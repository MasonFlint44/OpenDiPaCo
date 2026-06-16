"""Declarative cluster configuration for the ``opendipaco`` launcher.

One config file describes a whole run -- the model, the DiLoCo hyper-parameters, the
data source, the transport (host/port, auth, TLS, metrics), and the run schedule --
so every role (coordinator / scheduler / parameter server / worker) reads the *same*
file and agrees on shapes, ports, and secrets. Load it with :func:`load_config`
(YAML, TOML, or JSON by extension) and turn the model/diloco sections into the core
dataclasses with the ``*_config`` builders.

The schema is intentionally flat: each section is a small dataclass with defaults, so
a minimal file (even ``{}``) is valid and you only override what you need.
"""

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from ..config import BackboneConfig, DiLoCoConfig, DiPaCoConfig


@dataclass
class ModelCfg:
    vocab_size: int = 32000
    hidden_size: int = 512
    num_attention_heads: int = 8
    num_key_value_heads: int | None = None
    intermediate_size: int = 1024
    max_position_embeddings: int = 1024
    rope_theta: float = 10000.0
    layers_per_level: list[int] = field(default_factory=lambda: [2, 2])
    level_sizes: list[int] = field(default_factory=lambda: [4, 4])
    embedding: str = "shared"
    head: str = "shared"
    tie_word_embeddings: bool = False
    sequence_length: int = 256
    eval_sequence_length: int | None = None


@dataclass
class DiLoCoCfg:
    inner_steps: int = 50
    inner_lr: float = 4e-4
    inner_weight_decay: float = 0.1
    inner_grad_clip: float | None = 1.0
    inner_lr_schedule: str = "cosine"
    inner_warmup_steps: int = 0
    outer_lr: float = 0.7
    outer_momentum: float = 0.9
    outer_nesterov: bool = True
    rescale_by_sqrt_sharing: bool = True
    # bf16 mixed-precision inner loop (params/grads stay fp32; no loss scaling).
    inner_autocast: bool = False
    # Activation checkpointing (W3b): recompute body-block activations in backward
    # for a large activation-memory cut at ~one extra forward. Bit-exact, so it's
    # default-on for real runs (turn off only to trade memory back for speed).
    activation_checkpoint: bool = True
    # Chunked cross-entropy over this many token-chunks (W3c): avoids the full
    # [tokens, vocab] logits tensor -- set > 1 for a large vocab. 1 = off (the loss
    # sum order shifts ~1e-7 when on, so it's opt-in rather than default).
    loss_chunks: int = 1
    # Lossy VRAM levers (W3d; §0f-gated, off by default): int8 AdamW moments
    # (~4x optimizer cut) and aliasing the worker's private embed/head (saves the
    # copy, changes warm-round private dynamics). Validate before enabling.
    optim_8bit: bool = False
    dedup_private: bool = False


@dataclass
class DataCfg:
    source: str = "synthetic"          # "c4" | "synthetic"
    num_documents: int = 2000
    max_doc_tokens: int | None = 256
    min_doc_tokens: int = 1
    tokenizer: str | None = None       # name/path; None -> a sensible default per source
    cache_path: str | None = None      # single-file doc cache (load_c4_documents)
    shard_cache_dir: str | None = None # directory for sharded resumable ingestion
    routing: str = "kmeans"            # "kmeans" | "round_robin"
    router_seed: int = 0
    # "bytes": the server holds the corpus and ships packed shards (default).
    # "spec": the server ships a shard *recipe*; workers materialize shards
    # locally from the public source (data/spec.py). No per-path val split.
    ship: str = "bytes"
    synthetic_topics: int = 4
    synthetic_doc_len: int = 80


@dataclass
class TLSCfg:
    enabled: bool = False
    certfile: str | None = None
    keyfile: str | None = None
    cafile: str | None = None
    insecure: bool = False              # client: encrypt without verifying the server
    require_client_cert: bool = False   # server: mutual TLS
    server_hostname: str | None = None  # client: name to verify / SNI


@dataclass
class TransportCfg:
    # Connection substrate (W1; docs/w1-nat-design.md). "tcp" (default) is the
    # raw-socket reactor -- the deterministic anchor, bit-identical to pre-W1.
    # "libp2p" runs our wire frames over libp2p Noise streams (NAT traversal via
    # Circuit Relay v2 + DCUtR); needs the optional ``[nat]`` extra and a
    # ``transport.identity_key`` (the libp2p host key derives from it).
    kind: str = "tcp"                   # "tcp" | "libp2p"
    # libp2p listen multiaddrs (kind == "libp2p"); 0 picks an ephemeral port.
    libp2p_listen: list[str] = field(default_factory=lambda: ["/ip4/0.0.0.0/tcp/0"])
    # Relay multiaddrs a NAT'd peer reserves a forwarding slot on (k>=2 for
    # failover, D6); empty for a public peer that needs no relay.
    relays: list[str] = field(default_factory=list)
    dcutr: bool = True                  # attempt relayed->direct hole-punch upgrade (D9)
    # The coordinator/scheduler multiaddr a libp2p worker dials (kind ==
    # "libp2p"): a direct ``/ip4/.../p2p/<id>`` or a ``/p2p-circuit`` addr for a
    # NAT'd scheduler. (TCP uses ``connect_host``/``port``.)
    connect_libp2p: str | None = None
    host: str = "0.0.0.0"               # bind address (servers)
    connect_host: str | None = None     # address workers dial (defaults from host)
    port: int = 29500
    auth_key: str | None = None
    accept_keys: list[str] = field(default_factory=list)
    # Per-peer Ed25519 identity (Phase 1). ``identity_key`` is this node's
    # private-key PEM (see `opendipaco gen-identity`); when set, workers
    # authenticate by signing the server's challenge instead of HMAC.
    # ``admitted_peers`` lists the public keys (hex) a server accepts.
    identity_key: str | None = None
    admitted_peers: list[str] = field(default_factory=list)
    # Rendezvous ownership: the scheduler's public key (hex). Owners verify
    # epoch records and Ed25519 commit grants against it.
    scheduler_pub: str | None = None
    # Sharded mode: secret shared by the scheduler + parameter servers (NOT workers)
    # that signs commit grants, so workers can't forge push weights.
    grant_key: str | None = None
    max_msg_bytes: int | None = None
    # Home-grade (W4d, design D2): a dropped worker's lease re-leases sooner.
    # Workers heartbeat every heartbeat_interval (3s), so 20s = ~6 missed beats
    # before reclaim -- a slow-but-alive worker isn't reclaimed mid-task.
    heartbeat_timeout: float = 20.0
    heartbeat_interval: float = 3.0
    staleness_bound: int | None = None
    staleness_weight: str = "inverse"
    # Servers always reject non-finite contributions; this additionally clips a
    # pseudo-gradient whose L2 norm exceeds the cap (None = no cap).
    max_update_norm: float | None = None
    # Wire compression: "none" (fp32, default), "bf16" (2x), "int8" (bf16 weights
    # + int8 pseudo-gradients with error feedback, ~4x up), or "int4" (int4
    # per-group pseudo-gradients/deltas, ~8x up; bf16 weights). int8/int4 also set
    # the down-delta precision (W2a/W2c). Off (none) is byte-identical.
    compress: str = "none"
    # Downlink (weights) policy (W2; docs/w2-bandwidth-design.md): "full"
    # (default, byte-identical) re-ships full weights; "delta" ships int8
    # current-minus-keyframe when the worker holds a recent keyframe (the async
    # cache the version churn defeats). Changes numerics -> off by default;
    # validate with examples/validate_dynamics.py. Set on the scheduler AND owners.
    down: str = "full"
    # Up-path (pseudo-gradient) structured sparsification (W2b): the worker keeps
    # each gradient's top `up_density` fraction (per output-row for 2-D weights)
    # and error-feeds the dropped mass. 1.0 (default) = dense = byte-identical.
    # Changes numerics -> validate with examples/validate_dynamics.py. Sharded only.
    up_density: float = 1.0
    # When set, idle replies tell workers to wait this many seconds before
    # polling again (server-paced; otherwise workers use their own tight poll).
    idle_backoff: float | None = None
    metrics_port: int | None = None
    metrics_host: str = "0.0.0.0"
    metrics_log_interval: float = 0.0


@dataclass
class ShardedCfg:
    num_shards: int = 2
    # How workers/scheduler reach each parameter server: [[host, port], ...].
    parameter_servers: list[list] = field(default_factory=list)


@dataclass
class TrackerCfg:
    host: str = "0.0.0.0"
    connect_host: str | None = None    # address peers dial (defaults from host)
    port: int = 29600
    # Home-grade default (W4d, design D2): a rebooting consumer node should be
    # detectable in tens of seconds, not minutes. The library Tracker default
    # stays 120 (cluster) for the in-process anchor + unit tests.
    ttl: float = 30.0                  # registrations expire unless re-registered
    open_enrollment: bool = False      # True: any validly-signed record may register
    enroll_peers: list[str] = field(default_factory=list)  # pubkeys allowed to register


@dataclass
class OwnershipCfg:
    """Dynamic module ownership (Phase 2; ``docs/phase2-design.md``).

    ``static`` keeps today's fixed ``assign_shards`` parameter servers.
    ``rendezvous`` derives ownership from tracker liveness: owners register
    with the tracker, the scheduler signs owner-set epochs (HRW placement,
    ``k`` replicas per key, primary-only writes, pull replication), and
    failover is automatic. Rendezvous mode needs the ``tracker`` section, a
    scheduler ``transport.identity_key``, and ``transport.scheduler_pub`` on
    the owners/parameter servers.
    """

    # Home-grade detection timings (W4d, design D2): tuned for consumer churn
    # (machines sleep/reboot/drop), not cluster stability. Invariants kept:
    # owner_grace >= 2*tracker.ttl (a flapping owner mustn't thrash ownership)
    # and heartbeat_interval < tracker.ttl (one missed beat isn't an eviction).
    # The library EpochManager/ParameterServer defaults stay conservative
    # (240/60/30) for the in-process anchor + unit tests.
    mode: str = "static"               # "static" | "rendezvous"
    k: int = 3                         # replicas per module key (rank 0 = primary)
    salt: str = ""                     # run-level placement salt (changing it remaps all)
    bank_seed: int = 0                 # shared init seed: (0,0) = same bytes everywhere
    replicate_interval: float = 10.0   # backup pull cadence = the abrupt-failover loss window
    owner_grace: float = 60.0          # owner unseen this long -> dropped next epoch (>= 2*ttl)
    min_epoch_interval: float = 20.0   # at most one epoch bump per this many seconds
    epoch_poll_interval: float = 3.0   # scheduler's tracker-directory poll cadence
    heartbeat_interval: float = 10.0   # owner -> tracker re-registration (keep < ttl)
    advertise_host: str | None = None  # address other peers dial for this owner


@dataclass
class RobustnessCfg:
    """Byzantine-robustness (Phase 3; ``docs/phase3-design.md``). All off by
    default — ``mode: off`` keeps the run bit-identical to Phase 2.

    Turning ``mode: on`` enables owner-side **robust aggregation** of shared
    modules and the reputation/rate-limit gates; ``redundancy_rate > 0`` adds
    **redundant execution** (sampled tasks re-run and cross-checked, feeding
    reputation); ``private_policy: proposal`` makes private-module pushes
    proposals that apply only on agreement. These change training dynamics and
    must be validated against the deterministic anchor (plan §1.4;
    ``examples/validate_robustness.py``).

    **Liveness requirement for** ``private_policy: proposal``: a private module
    advances only when ``private_quorum`` scheduler-assigned checkers
    corroborate it, and checkers come from *surplus* workers. So it needs
    **worker oversupply** (more workers than paths) **and**
    ``private_quorum <= redundancy``; otherwise private modules (embedding/head)
    silently *stall* — the run still completes (commits advance the clock), but
    those modules never train. Use ``proposal`` only with surplus workers, or
    keep ``private_policy: overwrite`` (the default).
    """

    mode: str = "off"                  # "off" | "on" (robust aggregation + gates)
    aggregate: str = "trimmed_mean"    # "trimmed_mean" | "median" | "mean"
    quorum_target: int = 3             # contributions to buffer before aggregating
    quorum_timeout: float = 30.0       # flush a partial buffer after this many seconds
    # Redundant execution.
    redundancy: int = 3                # replicas per audited task (1 primary + checkers)
    redundancy_rate: float = 0.0       # fraction of tasks audited (0 = off)
    audit_timeout: float = 60.0        # resolve an audit after this long if incomplete
    version_history: int = 1           # owner retained versions (>1 to enable pinned checks)
    # Reputation + rate limiting.
    reputation_floor: float = 0.5      # fresh peers start here (Sybil: earn above it)
    reputation_credit: float = 0.02
    reputation_debit: float = 0.2
    reputation_halflife: float = 3600.0
    min_owner_reputation: float = 0.25  # below this -> demoted from the owner set
    rate_capacity: float = 8.0          # token bucket size (scaled by reputation)
    rate_refill_per_sec: float = 2.0
    # Private modules.
    private_policy: str = "overwrite"   # "overwrite" | "proposal"
    private_quorum: int = 2             # agreeing peers needed to apply a private proposal

    def __post_init__(self):
        # Catch the *guaranteed*-stall private-policy misconfigurations at load
        # time (a private module could never reach quorum), rather than letting
        # embedding/head silently freeze at runtime. The oversupply requirement
        # is runtime-only and stays documented on the class.
        if self.private_policy == "proposal":
            if self.redundancy < 2:
                raise ValueError(
                    "private_policy: proposal needs redundancy >= 2 (a primary + "
                    "at least one checker to corroborate); else private modules stall")
            if self.private_quorum > self.redundancy:
                raise ValueError(
                    f"private_quorum ({self.private_quorum}) > redundancy "
                    f"({self.redundancy}): a private proposal could never reach quorum")


@dataclass
class ScheduleCfg:
    """Control-plane topology (Phase 4; ``docs/phase4-design.md``).

    ``central`` (default) keeps today's single :class:`Scheduler` node: the
    global ``_T`` clock, the lease queue, scheduler-signed grants, and the
    scheduler as the run's trust root — bit-identical to Phase 3.

    ``decentralized`` removes the scheduler as a node. Work assignment becomes
    leaderless (HRW over ``(path, generation)`` and the live worker set, with
    deterministic takeover on lease expiry), the clock becomes the per-module
    ``(epoch, counter)`` version vectors, grants are minted by each path's
    primary **owner**, reputation/audits/rate-limits shard onto the owner tier,
    owners cross-check each other (quorum reads + replicated-aggregation digest
    agreement) so a minority of Byzantine owners is tolerated, and the tracker
    degrades to a bootstrap seed (owners gossip the directory). It *implies*
    ``ownership: rendezvous`` (it is built on the replicated owner tier) and
    changes training dynamics, so it must be validated against the anchor like
    every other dynamics change (plan §1.4).
    """

    mode: str = "central"              # "central" | "decentralized"
    lease_ttl: float = 20.0            # rank-0's window before takeover-on-expiry (home-grade, W4d)
    gossip_interval: float = 10.0      # owner-to-owner directory pull cadence
    read_quorum: int = 2               # replicas a fetch cross-checks (Byzantine reads)

    def __post_init__(self):
        if self.mode not in ("central", "decentralized"):
            raise ValueError(
                f"schedule.mode must be 'central' or 'decentralized', got {self.mode!r}")


@dataclass
class RunCfg:
    generations: int = 10
    batch_size: int = 8
    seed: int = 0
    device: str = "cpu"
    checkpoint_dir: str | None = None
    checkpoint_every: int = 0
    resume: bool = False
    max_tasks: int | None = None
    local_workers: int = 2             # workers the all-in-one `run` command spawns
    # Worker-advertised batch cap: the server clamps this worker's task batch
    # size to it (small-VRAM volunteers train smaller batches instead of OOMing).
    worker_max_batch: int | None = None
    # W5 task sizing (sharded mode; docs/w5-task-sizing-design.md). None (default)
    # = off, byte-identical: every task is the configured size. When set, the
    # scheduler sizes each task from the worker's measured rate so its lease lands
    # in ~task_seconds (batch first, then inner_steps; shrink-only). Changes
    # training dynamics -> validate with examples/validate_dynamics.py. A worker
    # too slow even for the minimum task (> task_seconds * park_factor, or below
    # min_task_rate tokens/s) is parked so it can't straggle a module.
    task_seconds: float | None = None
    park_factor: float = 3.0
    min_task_rate: float | None = None


@dataclass
class LaunchConfig:
    mode: str = "coordinator"          # "coordinator" | "sharded"
    model: ModelCfg = field(default_factory=ModelCfg)
    diloco: DiLoCoCfg = field(default_factory=DiLoCoCfg)
    data: DataCfg = field(default_factory=DataCfg)
    transport: TransportCfg = field(default_factory=TransportCfg)
    tls: TLSCfg = field(default_factory=TLSCfg)
    sharded: ShardedCfg = field(default_factory=ShardedCfg)
    tracker: TrackerCfg = field(default_factory=TrackerCfg)
    ownership: OwnershipCfg = field(default_factory=OwnershipCfg)
    robustness: RobustnessCfg = field(default_factory=RobustnessCfg)
    schedule: ScheduleCfg = field(default_factory=ScheduleCfg)
    run: RunCfg = field(default_factory=RunCfg)

    _SECTIONS = {  # name -> dataclass (class attr, not a field)
        "model": ModelCfg, "diloco": DiLoCoCfg, "data": DataCfg,
        "transport": TransportCfg, "tls": TLSCfg, "sharded": ShardedCfg,
        "tracker": TrackerCfg, "ownership": OwnershipCfg,
        "robustness": RobustnessCfg, "schedule": ScheduleCfg, "run": RunCfg,
    }

    @classmethod
    def from_dict(cls, d: dict | None) -> "LaunchConfig":
        d = dict(d or {})
        kw = {"mode": d.pop("mode", "coordinator")}
        for name, dc in cls._SECTIONS.items():
            kw[name] = _build_section(dc, d.pop(name, {}))
        if d:
            raise ValueError(f"unknown top-level config keys: {sorted(d)}")
        if kw["mode"] not in ("coordinator", "sharded"):
            raise ValueError(f"mode must be 'coordinator' or 'sharded', got {kw['mode']!r}")
        if kw["transport"].kind not in ("tcp", "libp2p"):
            raise ValueError(
                f"transport.kind must be 'tcp' or 'libp2p', got {kw['transport'].kind!r}")
        if kw["transport"].down not in ("full", "delta"):
            raise ValueError(
                f"transport.down must be 'full' or 'delta', got {kw['transport'].down!r}")
        # Delta-down lives on the sharded owner tier (the version ring + owner
        # fetch). The single-node coordinator has no such path, so down="delta"
        # there would silently do nothing -- fail fast instead.
        if kw["transport"].down == "delta" and kw["mode"] != "sharded":
            raise ValueError("transport.down: delta requires mode: sharded "
                             "(delta-down is served by the owner tier)")
        if not 0.0 < kw["transport"].up_density <= 1.0:
            raise ValueError("transport.up_density must be in (0, 1], got "
                             f"{kw['transport'].up_density!r}")
        if kw["transport"].up_density < 1.0 and kw["mode"] != "sharded":
            raise ValueError("transport.up_density < 1.0 requires mode: sharded "
                             "(it is stamped on sharded tasks)")
        # Decentralized scheduling is built on the replicated owner tier, so it
        # requires rendezvous ownership (Phase 4 D9). Catch the mismatch at load
        # rather than half-wiring a run with no owners to mint grants.
        if kw["schedule"].mode == "decentralized" and kw["ownership"].mode != "rendezvous":
            raise ValueError(
                "schedule.mode: decentralized requires ownership.mode: rendezvous "
                "(it builds on the replicated owner tier)")
        return cls(**kw)

    def connect_addr(self) -> tuple[str, int]:
        """Address a worker dials for the coordinator/scheduler."""
        t = self.transport
        host = t.connect_host or (t.host if t.host not in ("0.0.0.0", "::") else "127.0.0.1")
        return host, t.port

    def tracker_connect_addr(self) -> tuple[str, int]:
        """Address peers dial for the tracker."""
        t = self.tracker
        host = t.connect_host or (t.host if t.host not in ("0.0.0.0", "::") else "127.0.0.1")
        return host, t.port


def _build_section(dc, data: dict):
    data = dict(data or {})
    names = {f.name for f in dataclasses.fields(dc)}
    unknown = set(data) - names
    if unknown:
        raise ValueError(f"unknown keys for [{dc.__name__}]: {sorted(unknown)}")
    return dc(**data)


def load_config(path) -> LaunchConfig:
    """Parse a cluster config file (``.yaml``/``.yml``, ``.toml``, or ``.json``)."""
    path = Path(path)
    text = path.read_text()
    suffix = path.suffix.lower()
    if suffix in (".yaml", ".yml"):
        try:
            import yaml
        except ImportError as e:  # pragma: no cover
            raise ImportError("YAML config needs PyYAML: pip install 'opendipaco[launch]'") from e
        data = yaml.safe_load(text)
    elif suffix == ".toml":
        import tomllib
        data = tomllib.loads(text)
    elif suffix == ".json":
        import json
        data = json.loads(text)
    else:
        raise ValueError(f"unsupported config format '{suffix}'; use .yaml, .toml or .json")
    return LaunchConfig.from_dict(data)


# -- builders: config sections -> core dataclasses ---------------------------


def backbone_config(m: ModelCfg) -> BackboneConfig:
    return BackboneConfig(
        vocab_size=m.vocab_size, hidden_size=m.hidden_size,
        num_attention_heads=m.num_attention_heads, num_key_value_heads=m.num_key_value_heads,
        intermediate_size=m.intermediate_size, max_position_embeddings=m.max_position_embeddings,
        rope_theta=m.rope_theta, layers_per_level=list(m.layers_per_level),
    )


def dipaco_config(m: ModelCfg) -> DiPaCoConfig:
    return DiPaCoConfig(
        backbone=backbone_config(m), level_sizes=list(m.level_sizes),
        embedding=m.embedding, head=m.head, tie_word_embeddings=m.tie_word_embeddings,
        sequence_length=m.sequence_length, eval_sequence_length=m.eval_sequence_length,
    )


def diloco_config(d: DiLoCoCfg) -> DiLoCoConfig:
    return DiLoCoConfig(
        inner_steps=d.inner_steps, inner_lr=d.inner_lr, inner_weight_decay=d.inner_weight_decay,
        inner_grad_clip=d.inner_grad_clip, inner_lr_schedule=d.inner_lr_schedule,
        inner_warmup_steps=d.inner_warmup_steps, outer_lr=d.outer_lr,
        outer_momentum=d.outer_momentum, outer_nesterov=d.outer_nesterov,
        rescale_by_sqrt_sharing=d.rescale_by_sqrt_sharing,
        inner_autocast=d.inner_autocast, activation_checkpoint=d.activation_checkpoint,
        loss_chunks=d.loss_chunks, optim_8bit=d.optim_8bit, dedup_private=d.dedup_private,
    )
