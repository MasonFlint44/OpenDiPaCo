from opendipaco.topology import PathTopology


def test_path_indexing_roundtrip():
    topo = PathTopology((2, 3, 4))
    assert topo.num_paths == 24
    for i, path in enumerate(topo.paths()):
        assert topo.path_index(path) == i
        assert topo.path_from_index(i) == path


def test_module_keys_and_sharing():
    topo = PathTopology((2, 3))
    keys = topo.module_keys()
    assert "embed" in keys and "head" in keys
    assert "L0E0" in keys and "L1E2" in keys
    # Each expert at level 0 is shared by num_paths / 2 = 3 paths.
    assert topo.sharing_count("L0E0") == 3
    assert topo.sharing_count("L1E0") == 2
    # Shared embed/head touch every path.
    assert topo.sharing_count("embed") == topo.num_paths
    assert len(topo.paths_through_module("L0E1")) == 3
    assert all(p[0] == 1 for p in topo.paths_through_module("L0E1"))


def test_path_module_keys_order():
    topo = PathTopology((2, 2))
    assert topo.path_module_keys((1, 0)) == ["embed", "L0E1", "L1E0", "head"]


def test_private_embed_head():
    topo = PathTopology((2, 2), embedding="private", head="private")
    keys = topo.module_keys()
    # No shared embed/head; one private copy per path instead.
    assert "embed" not in keys and "head" not in keys
    assert {"embed.p0", "embed.p1", "embed.p2", "embed.p3"} <= set(keys)
    # Private modules belong to a single path and are never averaged.
    assert topo.sharing_count("embed.p1") == 1
    assert topo.paths_through_module("embed.p1") == [topo.path_from_index(1)]
    # Path (0,1) has index 1; its private embed/head carry that suffix.
    assert topo.path_module_keys((0, 1)) == ["embed.p1", "L0E0", "L1E1", "head.p1"]


def test_private_trunk_block_via_segments():
    from opendipaco.topology import Segment, Sharing

    segs = [
        Segment("embed", 0, 1, Sharing.SHARED),
        Segment("body", 1, 1, Sharing.PRIVATE),   # private trunk block (per path)
        Segment("body", 2, 2, Sharing.SHARED),    # routing level of 2 experts
        Segment("head", 0, 1, Sharing.SHARED),
    ]
    topo = PathTopology(segments=segs)
    assert topo.num_paths == 2
    keys = topo.module_keys()
    assert {"B1.p0", "B1.p1"} <= set(keys)   # private block, one per path
    assert {"L0E0", "L0E1"} <= set(keys)     # the shared routing level
    assert topo.sharing_count("B1.p0") == 1
    assert topo.sharing_count("L0E0") == 1   # 2 paths / 2 experts
    # Layer offsets: private block occupies layer 0, the routing expert layers 1-2.
    assert topo.offset_of_key("B1.p0") == 0
    assert topo.offset_of_key("L0E0") == 1
