use core::arch::wasm32::v128;

use crate::arch::{all::packedpair::Pair, generic::packedpair};

#[derive(Clone, Debug)]
pub struct Finder(packedpair::Finder<v128>);

impl Finder {
    #[inline]
    pub fn new(needle: &[u8]) -> Option<Finder> {
        Finder::with_pair(needle, Pair::new(needle)?)
    }

    #[inline]
    pub fn with_pair(needle: &[u8], pair: Pair) -> Option<Finder> {
        #[cfg(target_feature = "simd128")]
        {
            // SAFETY: we check that simd128 is available above. We are also
            // guaranteed to have needle.len() > 1 because we have a valid
            // Pair.
            unsafe { Some(Finder::with_pair_impl(needle, pair)) }
        }
        #[cfg(not(target_feature = "simd128"))]
        {
            None
        }
    }

    /// Create a new `Finder` specific to wasm32 v128 vectors and routines.
    ///
    /// # Safety
    ///
    /// Same as the safety for `packedpair::Finder::new`.
    #[target_feature(enable = "simd128")]
    #[inline]
    unsafe fn with_pair_impl(needle: &[u8], pair: Pair) -> Finder {
        let finder = packedpair::Finder::<v128>::new(needle, pair);
        Finder(finder)
    }

    /// Execute a search using wasm32 v128 vectors and routines.
    ///
    /// # Panics
    ///
    /// When `haystack.len()` is less than [`Finder::min_haystack_len`].
    #[inline]
    pub fn find(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        self.find_impl(haystack, needle)
    }

    /// Execute a search using wasm32 v128 vectors and routines.
    ///
    /// # Panics
    ///
    /// When `haystack.len()` is less than [`Finder::min_haystack_len`].
    #[inline]
    pub fn find_prefilter(&self, haystack: &[u8]) -> Option<usize> {
        self.find_prefilter_impl(haystack)
    }

    /// Execute a search using wasm32 v128 vectors and routines.
    ///
    /// # Panics
    ///
    /// When `haystack.len()` is less than [`Finder::min_haystack_len`].
    ///
    /// # Safety
    ///
    /// (The target feature safety obligation is automatically fulfilled by
    /// virtue of being a method on `Finder`, which can only be constructed
    /// when it is safe to call `simd128` routines.)
    #[target_feature(enable = "simd128")]
    #[inline]
    fn find_impl(&self, haystack: &[u8], needle: &[u8]) -> Option<usize> {
        // SAFETY: The target feature safety obligation is automatically
        // fulfilled by virtue of being a method on `Finder`, which can only be
        // constructed when it is safe to call `simd128` routines.
        unsafe { self.0.find(haystack, needle) }
    }

    /// Execute a prefilter search using wasm32 v128 vectors and routines.
    ///
    /// # Panics
    ///
    /// When `haystack.len()` is less than [`Finder::min_haystack_len`].
    ///
    /// # Safety
    ///
    /// (The target feature safety obligation is automatically fulfilled by
    /// virtue of being a method on `Finder`, which can only be constructed
    /// when it is safe to call `simd128` routines.)
    #[target_feature(enable = "simd128")]
    #[inline]
    fn find_prefilter_impl(&self, haystack: &[u8]) -> Option<usize> {
        // SAFETY: The target feature safety obligation is automatically
        // fulfilled by virtue of being a method on `Finder`, which can only be
        // constructed when it is safe to call `simd128` routines.
        unsafe { self.0.find_prefilter(haystack) }
    }

    /// Returns the pair of offsets (into the needle) used to check as a
    /// predicate before confirming whether a needle exists at a particular
    /// position.
    #[inline]
    pub fn pair(&self) -> &Pair {
        self.0.pair()
    }

    /// Returns the minimum haystack length that this `Finder` can search.
    ///
    /// Using a haystack with length smaller than this in a search will result
    /// in a panic. The reason for this restriction is that this finder is
    /// meant to be a low-level component that is part of a larger substring
    /// strategy. In that sense, it avoids trying to handle all cases and
    /// instead only handles the cases that it can handle very well.
    #[inline]
    pub fn min_haystack_len(&self) -> usize {
        self.0.min_haystack_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn find(haystack: &[u8], needle: &[u8]) -> Option<Option<usize>> {
        let f = Finder::new(needle)?;
        if haystack.len() < f.min_haystack_len() {
            return None;
        }
        Some(f.find(haystack, needle))
    }

    define_substring_forward_quickcheck!(find);

    #[test]
    fn forward_substring() {
        crate::tests::substring::Runner::new().fwd(find).run()
    }

    #[test]
    fn forward_packedpair() {
        fn find(
            haystack: &[u8],
            needle: &[u8],
            index1: u8,
            index2: u8,
        ) -> Option<Option<usize>> {
            let pair = Pair::with_indices(needle, index1, index2)?;
            let f = Finder::with_pair(needle, pair)?;
            if haystack.len() < f.min_haystack_len() {
                return None;
            }
            Some(f.find(haystack, needle))
        }
        crate::tests::packedpair::Runner::new().fwd(find).run()
    }

    #[test]
    fn forward_packedpair_prefilter() {
        fn find(
            haystack: &[u8],
            needle: &[u8],
            index1: u8,
            index2: u8,
        ) -> Option<Option<usize>> {
            let pair = Pair::with_indices(needle, index1, index2)?;
            let f = Finder::with_pair(needle, pair)?;
            if haystack.len() < f.min_haystack_len() {
                return None;
            }
            Some(f.find_prefilter(haystack))
        }
        crate::tests::packedpair::Runner::new().fwd(find).run()
    }
}
