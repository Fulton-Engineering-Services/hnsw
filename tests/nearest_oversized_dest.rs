//! Regression tests for the destination-slice bug in `Hnsw::nearest` /
//! `Hnsw::search_layer`.
//!
//! Upstream `v0.11.0` computed `found = min(dest.len(), searcher.nearest.len())`
//! and then did `dest.copy_from_slice(&searcher.nearest[..found])`. Copying into
//! the *whole* `dest` requires `found == dest.len()`, so any call where the search
//! could not fill the destination panicked with:
//!
//! ```text
//! copy_from_slice: source slice length (129) does not match destination slice length (200)
//! ```
//!
//! The result pool width is query-dependent: `search_zero_layer` only pushes a
//! node onto the candidate queue when that node entered the result pool, so the
//! traversal stops as soon as no candidate can improve on the pool. `dest` may
//! therefore legitimately be longer than whatever the search was able to collect,
//! for reasons the caller cannot observe (`Searcher::nearest` is private and no
//! neighbour accessor is public).
//!
//! These tests build a graph with far fewer nodes than the requested destination
//! width, which makes `pool_len < dest.len()` certain rather than data-dependent.

use hnsw::{Hnsw, Searcher};
use rand_pcg::Pcg64;
use space::{Metric, Neighbor};

struct Euclidean;

impl Metric<&[f64]> for Euclidean {
    type Unit = u64;

    fn distance(&self, a: &&[f64], b: &&[f64]) -> u64 {
        a.iter()
            .zip(b.iter())
            .map(|(x, y)| {
                let d = x - y;
                // Integer-valued features keep this exact.
                (d * d) as u64
            })
            .sum()
    }
}

const FEATURES: &[&[f64]] = &[
    &[0.0, 0.0, 0.0],
    &[1.0, 0.0, 0.0],
    &[0.0, 1.0, 0.0],
    &[0.0, 0.0, 1.0],
    &[1.0, 1.0, 0.0],
    &[3.0, 3.0, 3.0],
];

fn test_hnsw() -> (
    Hnsw<Euclidean, &'static [f64], Pcg64, 12, 24>,
    Searcher<u64>,
) {
    let mut searcher = Searcher::default();
    let mut hnsw = Hnsw::new(Euclidean);
    for &feature in FEATURES {
        hnsw.insert(feature, &mut searcher);
    }
    (hnsw, searcher)
}

fn dest_of(len: usize) -> Vec<Neighbor<u64>> {
    vec![
        Neighbor {
            index: !0,
            distance: 0
        };
        len
    ]
}

/// The bug: asking for more results than the graph can possibly yield panicked.
///
/// Six nodes exist, so the result pool can never exceed six, yet `dest` is 512
/// long and `ef` is 512. Unpatched this panics in `copy_from_slice`; patched it
/// must return the six reachable neighbours.
#[test]
fn oversized_destination_returns_the_available_neighbours() {
    let (hnsw, mut searcher) = test_hnsw();
    let mut dest = dest_of(512);

    let found = hnsw.nearest(&&[3.0, 3.0, 3.0][..], 512, &mut searcher, &mut dest);

    assert!(
        !found.is_empty(),
        "a non-empty graph must yield at least one neighbour"
    );
    assert!(
        found.len() <= FEATURES.len(),
        "returned {} neighbours from a {} node graph",
        found.len(),
        FEATURES.len()
    );
    assert_eq!(
        found[0].index, 5,
        "the exact match must be first, got {:?}",
        found[0]
    );
    assert_eq!(found[0].distance, 0, "the exact match must have distance 0");
    for pair in found.windows(2) {
        assert!(
            pair[0].distance <= pair[1].distance,
            "results must stay sorted by distance: {:?}",
            found
        );
    }
}

/// A destination wider than `ef` must be clamped to what the pool can hold rather
/// than panicking. This is the shape that reached us from `memvid-core`, whose
/// search passed `dest.len() = limit` while the pool was bounded by `ef`.
#[test]
fn destination_wider_than_ef_does_not_panic() {
    let (hnsw, mut searcher) = test_hnsw();
    let mut dest = dest_of(1000);

    let found = hnsw.nearest(&&[0.0, 0.0, 0.0][..], 4, &mut searcher, &mut dest);

    assert!(
        found.len() <= 4,
        "ef = 4 must bound the result count, got {}",
        found.len()
    );
    assert_eq!(found[0].index, 0, "the exact match must be first");
}

/// `search_layer` with `level = 0` shares the same copy site and must behave the
/// same way.
#[test]
fn search_layer_with_oversized_destination_does_not_panic() {
    let (hnsw, mut searcher) = test_hnsw();
    let mut dest = dest_of(256);

    let found = hnsw.search_layer(&&[1.0, 1.0, 0.0][..], 256, 0, &mut searcher, &mut dest);

    assert!(!found.is_empty(), "level 0 must yield neighbours");
    assert!(found.len() <= FEATURES.len(), "got {}", found.len());
    assert_eq!(found[0].index, 4, "the exact match must be first");
}

/// Guards the fix itself: narrowing the copy must not corrupt the results. The
/// first `n` slots must hold the same neighbours an exact-width destination gets.
#[test]
fn results_match_an_exact_width_destination() {
    let (hnsw, mut searcher) = test_hnsw();

    let mut exact = dest_of(3);
    hnsw.nearest(&&[3.0, 3.0, 3.0][..], 3, &mut searcher, &mut exact);

    let (hnsw2, mut searcher2) = test_hnsw();
    let mut wide = dest_of(400);
    let found = hnsw2.nearest(&&[3.0, 3.0, 3.0][..], 3, &mut searcher2, &mut wide);

    assert_eq!(
        found,
        &exact[..found.len()],
        "a wide destination must agree with an exact-width one"
    );
}
