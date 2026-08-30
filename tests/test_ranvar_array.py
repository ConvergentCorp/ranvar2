import pickle

import pytest

import ranvar2 as mc


def makeArray():
    """Build a small RanVarArray of parametric digests to exercise."""
    return mc.RanVarArray.fromList([
        mc.Normal(100.0, 5.0),
        mc.Normal(200.0, 10.0),
        mc.NegBinom(5.0, 1.0),
    ])


def test_fromList_wraps_ranvars():
    """Tests that fromList builds an array of the given length and order."""
    arr = makeArray()

    assert len(arr) == 3
    assert [r.mean() for r in arr] == [100.0, 200.0, 5.0]


def test_constructor_accepts_any_iterable():
    """Tests that the plain constructor works the same way as fromList."""
    arr = mc.RanVarArray(mc.Normal(m, 1.0) for m in [1.0, 2.0, 3.0])

    assert len(arr) == 3


def test_empty_by_default():
    """Tests that an array built with no items starts empty."""
    arr = mc.RanVarArray()

    assert len(arr) == 0
    assert list(arr) == []
    assert bool(arr) is False


def test_rejects_non_ranvar_items():
    """Tests that anything but a RanVar is refused, on construction and append."""
    with pytest.raises(TypeError):
        mc.RanVarArray([1, 2, 3])

    arr = makeArray()

    with pytest.raises(TypeError):
        arr.append(5)

    with pytest.raises(TypeError):
        arr.insert(0, 'not a ranvar')

    with pytest.raises(TypeError):
        arr[0] = 5

    with pytest.raises(TypeError):
        arr.extend([mc.Normal(0, 1), 5])


def test_indexing_and_slicing():
    """Tests __getitem__ for both a single index and a slice."""
    arr = makeArray()

    assert arr[0].mean() == 100.0
    assert arr[-1].mean() == 5.0

    sliced = arr[0:2]

    assert isinstance(sliced, mc.RanVarArray)
    assert len(sliced) == 2
    assert sliced[1] is arr[1]


def test_setitem_and_delitem():
    """Tests replacing and removing elements by index."""
    arr = makeArray()
    replacement = mc.Normal(1.0, 1.0)

    arr[0] = replacement
    assert arr[0] is replacement

    del arr[0]
    assert len(arr) == 2
    assert arr[0].mean() == 200.0


def test_iteration_and_containment():
    """Tests __iter__, __reversed__ and __contains__."""
    arr = makeArray()
    items = list(arr)

    assert items == list(arr)
    assert list(reversed(arr)) == list(reversed(items))
    assert items[0] in arr
    assert mc.Normal(999.0, 1.0) not in arr


def test_append_extend_insert():
    """Tests growing the array with append, extend and insert."""
    arr = mc.RanVarArray()
    a, b, c = mc.Normal(1, 1), mc.Normal(2, 1), mc.Normal(3, 1)

    arr.append(a)
    arr.extend([b, c])
    arr.insert(0, mc.Normal(0, 1))

    assert [r.mean() for r in arr] == [0.0, 1.0, 2.0, 3.0]


def test_pop_remove_clear():
    """Tests shrinking the array with pop, remove and clear."""
    arr = makeArray()
    last = arr.pop()

    assert last.mean() == 5.0
    assert len(arr) == 2

    first = arr[0]
    arr.remove(first)
    assert first not in arr

    arr.clear()
    assert len(arr) == 0


def test_index_and_count():
    """Tests locating and counting elements."""
    arr = makeArray()
    target = arr[1]

    assert arr.index(target) == 1
    assert arr.count(target) == 1

    with pytest.raises(ValueError):
        arr.index(mc.Normal(999.0, 1.0))


def test_copy_is_independent():
    """Tests that copy() gives a separate array over the same elements."""
    arr = makeArray()
    other = arr.copy()

    other.pop()

    assert len(arr) == 3
    assert len(other) == 2
    assert arr[0] is other[0]


def test_sort_and_reverse():
    """Tests sort() with a key function and reverse()."""
    arr = mc.RanVarArray.fromList([mc.Normal(3, 1), mc.Normal(1, 1), mc.Normal(2, 1)])

    arr.sort(key=lambda r: r.mean())
    assert [r.mean() for r in arr] == [1.0, 2.0, 3.0]

    arr.reverse()
    assert [r.mean() for r in arr] == [3.0, 2.0, 1.0]


def test_concatenation_and_repetition():
    """Tests +, += and * against other RanVarArrays and plain lists."""
    a = mc.RanVarArray.fromList([mc.Normal(1, 1)])
    b = mc.RanVarArray.fromList([mc.Normal(2, 1)])

    combined = a + b
    assert isinstance(combined, mc.RanVarArray)
    assert [r.mean() for r in combined] == [1.0, 2.0]

    withList = a + [mc.Normal(3, 1)]
    assert [r.mean() for r in withList] == [1.0, 3.0]

    fromList = [mc.Normal(0, 1)] + a
    assert [r.mean() for r in fromList] == [0.0, 1.0]

    a += b
    assert [r.mean() for r in a] == [1.0, 2.0]

    repeated = mc.RanVarArray.fromList([mc.Normal(5, 1)]) * 3
    assert [r.mean() for r in repeated] == [5.0, 5.0, 5.0]


def test_equality():
    """Tests __eq__ against another RanVarArray and a plain list."""
    items = [mc.Normal(1, 1), mc.Normal(2, 1)]

    a = mc.RanVarArray(items)
    b = mc.RanVarArray(items)

    assert a == b
    assert a == items
    assert a != mc.RanVarArray([mc.Normal(9, 1)])
    assert (a == 'not an array') is False


def test_pickling_round_trips():
    """Tests that a RanVarArray survives pickle/unpickle with its elements."""
    arr = makeArray()
    restored = pickle.loads(pickle.dumps(arr))

    assert isinstance(restored, mc.RanVarArray)
    assert [r.mean() for r in restored] == [r.mean() for r in arr]
