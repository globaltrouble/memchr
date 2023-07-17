use core::arch::x86_64::__m128i;

use crate::{arch::generic::memchr as generic, ext::Pointer, vector::Vector};

/// Finds all occurrences of a single byte in a haystack.
#[derive(Clone, Debug)]
pub struct One(generic::One<__m128i>);

impl One {
    /// Create a new searcher that finds occurrences of the needle byte given.
    ///
    /// This particular searcher is specialized to use SSE2 vector instructions
    /// that typically make it quite fast.
    ///
    /// If SSE2 is unavailable in the current environment, then `None` is
    /// returned.
    #[inline]
    pub fn new(needle: u8) -> Option<One> {
        #[cfg(target_feature = "sse2")]
        {
            // SAFETY: we check that sse2 is available above. We are also
            // guaranteed to have needle.len() > 1 because we have a valid
            // Pair.
            unsafe { Some(One::new_unchecked(needle)) }
        }
        #[cfg(not(target_feature = "sse2"))]
        {
            None
        }
    }

    /// Create a new finder specific to SSE2 vectors and routines without
    /// checking that SSE2 is available.
    ///
    /// # Safety
    ///
    /// Callers must guarantee that it is safe to execute `sse2` instructions
    /// in the current environment.
    ///
    /// Note that it is a common misconception that if one compiles for an
    /// `x86_64` target, then they therefore automatically have access to SSE2
    /// instructions. While this is almost always the case, it isn't true in
    /// 100% of cases.
    #[target_feature(enable = "sse2")]
    #[inline]
    pub unsafe fn new_unchecked(needle: u8) -> One {
        One(generic::One::new(needle))
    }

    /// Return the first occurrence of one of the needle bytes in the given
    /// haystack. If no such occurrence exists, then `None` is returned.
    ///
    /// The occurrence is reported as an offset into `haystack`. Its maximum
    /// value for a non-empty haystack is `haystack.len() - 1`.
    #[inline]
    pub fn find(&self, haystack: &[u8]) -> Option<usize> {
        // SAFETY: `find_raw` guarantees that if a pointer is returned, it
        // falls within the bounds of the start and end pointers.
        unsafe {
            generic::search_slice_with_raw(haystack, |s, e| {
                self.find_raw(s, e)
            })
        }
    }

    /// Like `find`, but accepts and returns raw pointers.
    ///
    /// When a match is found, the pointer returned is guaranteed to be
    /// `>= start` and `< end`.
    ///
    /// This routine is useful if you're already using raw pointers and would
    /// like to avoid converting back to a slice before executing a search.
    ///
    /// # Safety
    ///
    /// * Both `start` and `end` must be valid for reads.
    /// * Both `start` and `end` must point to an initialized value.
    /// * Both `start` and `end` must point to the same allocated object and
    /// must either be in bounds or at most one byte past the end of the
    /// allocated object.
    /// * Both `start` and `end` must be _derived from_ a pointer to the same
    /// object.
    /// * The distance between `start` and `end` must not overflow `isize`.
    /// * The distance being in bounds must not rely on "wrapping around" the
    /// address space.
    ///
    /// Note that callers may pass a pair of pointers such that `start >= end`.
    /// In that case, `None` will always be returned.
    #[inline]
    pub unsafe fn find_raw(
        &self,
        start: *const u8,
        end: *const u8,
    ) -> Option<*const u8> {
        if end.distance(start) < __m128i::BYTES {
            // SAFETY: We require the caller to pass valid start/end pointers.
            return generic::fwd_byte_by_byte(start, end, |b| {
                b == self.0.needle1()
            });
        }
        // SAFETY: Building a `One` means it's safe to call 'sse2' routines.
        // Also, we've checked that our haystack is big enough to run on the
        // vector routine. Pointer validity is caller's responsibility.
        self.find_raw_impl(start, end)
    }

    /// Execute a search using SSE2 vectors and routines.
    ///
    /// # Safety
    ///
    /// Same as [`One::find_raw`], except the distance between `start` and
    /// `end` must be at least the size of an SSE2 vector (in bytes).
    ///
    /// (The target feature safety obligation is automatically fulfilled by
    /// virtue of being a method on `One`, which can only be constructed
    /// when it is safe to call `sse2` routines.)
    #[target_feature(enable = "sse2")]
    #[inline]
    unsafe fn find_raw_impl(
        &self,
        start: *const u8,
        end: *const u8,
    ) -> Option<*const u8> {
        self.0.find_raw(start, end)
    }

    /// Returns an iterator over all occurrences of the needle byte in the
    /// given haystack.
    ///
    /// The iterator returned implements `DoubleEndedIterator`. This means it
    /// can also be used to find occurrences in reverse order.
    #[inline]
    pub fn iter<'a, 'h>(&'a self, haystack: &'h [u8]) -> OneIter<'a, 'h> {
        OneIter { searcher: self, it: generic::Iter::new(haystack) }
    }
}

/// An iterator over all occurrences of a single byte in a haystack.
///
/// This iterator implements `DoubleEndedIterator`, which means it can also be
/// used to find occurrences in reverse order.
///
/// This iterator is created by the [`One::iter`] method.
///
/// The lifetime parameters are as follows:
///
/// * `'a` refers to the lifetime of the underlying [`One`] searcher.
/// * `'h` refers to the lifetime of the haystack being searched.
#[derive(Clone, Debug)]
pub struct OneIter<'a, 'h> {
    searcher: &'a One,
    it: generic::Iter<'h>,
}

impl<'a, 'h> Iterator for OneIter<'a, 'h> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        // SAFETY: We rely on the generic iterator to provide valid start
        // and end pointers, but we guarantee that any pointer returned by
        // 'find_raw' falls within the bounds of the start and end pointer.
        unsafe { self.it.next(|s, e| self.searcher.find_raw(s, e)) }
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        self.it.size_hint()
    }
}

impl<'a, 'h> DoubleEndedIterator for OneIter<'a, 'h> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        // SAFETY: We rely on the generic iterator to provide valid start
        // and end pointers, but we guarantee that any pointer returned by
        // 'rfind_raw' falls within the bounds of the start and end pointer.
        // unsafe { self.it.next_back(|s, e| self.searcher.rfind_raw(s, e)) }
        todo!()
    }
}

impl<'a, 'h> core::iter::FusedIterator for OneIter<'a, 'h> {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn forward_one() {
        crate::tests::memchr::Runner::new(1).forward(|haystack, needles| {
            Some(One::new(needles[0])?.iter(haystack).collect())
        })
    }
}
