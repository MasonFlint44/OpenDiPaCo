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
    heartbeat_timeout: float = 30.0
    heartbeat_interval: float = 3.0
    staleness_bound: int | None = None
    staleness_weight: str = "inverse"
    # Servers always reject non-finite contributions; this additionally clips a
    # pseudo-gradient whose L2 norm exceeds the cap (None = no cap).
    max_update_norm: float | None = None
    # Wire compression: "none" (fp32, default), "bf16" (2x), or "int8"
    # (bf16 weights + int8 pseudo-gradients with error feedback, ~4x up).
    compress: str = "none"
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
    ttl: float = 120.0                 # registrations expire unless re-registered
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

    mode: str = "static"               # "static" | "rendezvous"
    k: int = 3                         # replicas per module key (rank 0 = primary)
    salt: str = ""                     # run-level placement salt (changing it remaps all)
    bank_seed: int = 0                 # shared init seed: (0,0) = same bytes everywhere
    replicate_interval: float = 10.0   # backup pull cadence = the failover loss window
    owner_grace: float = 240.0         # owner unseen this long -> dropped next epoch
    min_epoch_interval: float = 60.0   # at most one epoch bump per this many seconds
    epoch_poll_interval: float = 5.0   # scheduler's tracker-directory poll cadence
    heartbeat_interval: float = 30.0   # owner -> tracker re-registration (keep < ttl)
    advertise_host: str | None = None  # address other peers dial for this owner


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
    run: RunCfg = field(default_factory=RunCfg)

    _SECTIONS = {  # name -> dataclass (class attr, not a field)
        "model": ModelCfg, "diloco": DiLoCoCfg, "data": DataCfg,
        "transport": TransportCfg, "tls": TLSCfg, "sharded": ShardedCfg,
        "tracker": TrackerCfg, "ownership": OwnershipCfg, "run": RunCfg,
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
        inner_autocast=d.inner_autocast,
    )
