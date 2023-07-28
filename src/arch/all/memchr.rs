/*!
This module defines architecture independent implementations of `memchr` and
friends.

The main types in this module are [`One`], [`Two`] and [`Three`]. They are for
searching for one, two or three distinct bytes, respectively, in a haystack.
Each type also has corresponding double ended iterators. These searchers
are typically slower than hand-coded vector routines accomplishing the same
task, but are also typically faster than naive scalar code. These routines
effectively work by treating a `usize` as a vector of 8-bit lanes, and thus
achieves some level of data parallelism even without explicit vector support.

Only one, two and three bytes are supported because three bytes is about
the point where one sees diminishing returns. Beyond this point and it's
probably (but not necessarily) better to just use a simple `[bool; 256]` array
or similar. However, it depends mightily on the specific work-load and the
expected match frequency.
*/

// BREADCRUMBS: I think at this point I'd like to add AVX2 (just One::find)
// and then hook-up rebar? Basically, I want to get this code under a profile
// now before doing too much more work in case shit needs to change.

use crate::{arch::generic::memchr as generic, ext::Pointer};

/// The number of bytes in a single `usize` value.
const USIZE_BYTES: usize = (usize::BITS / 8) as usize;
/// The bits that must be zero for a `*const usize` to be properly aligned.
const USIZE_ALIGN: usize = USIZE_BYTES - 1;

/// Finds all occurrences of a single byte in a haystack.
#[derive(Clone, Debug)]
pub struct One {
    s1: u8,
    v1: usize,
}

impl One {
    /// The number of bytes we examine per each iteration of our search loop.
    const LOOP_BYTES: usize = 2 * USIZE_BYTES;

    /// Create a new searcher that finds occurrences of the byte given.
    #[inline]
    pub fn new(needle: u8) -> One {
        One { s1: needle, v1: splat(needle) }
    }

    /// Return the first occurrence of the needle in the given haystack. If no
    /// such occurrence exists, then `None` is returned.
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

    /// Return the last occurrence of the needle in the given haystack. If no
    /// such occurrence exists, then `None` is returned.
    ///
    /// The occurrence is reported as an offset into `haystack`. Its maximum
    /// value for a non-empty haystack is `haystack.len() - 1`.
    #[inline]
    pub fn rfind(&self, haystack: &[u8]) -> Option<usize> {
        // SAFETY: `find_raw` guarantees that if a pointer is returned, it
        // falls within the bounds of the start and end pointers.
        unsafe {
            generic::search_slice_with_raw(haystack, |s, e| {
                self.rfind_raw(s, e)
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
        if start >= end {
            return None;
        }
        let confirm = |b| self.confirm(b);
        let len = end.distance(start);
        if len < USIZE_BYTES {
            return generic::fwd_byte_by_byte(start, end, confirm);
        }

        // The start of the search may not be aligned to `*const usize`,
        // so we do an unaligned load here.
        let chunk = start.cast::<usize>().read_unaligned();
        if self.has_needle(chunk) {
            return generic::fwd_byte_by_byte(start, end, confirm);
        }

        // And now we start our search at a guaranteed aligned position.
        // The first iteration of the loop below will overlap with the the
        // unaligned chunk above in cases where the search starts at an
        // unaligned offset, but that's okay as we're only here if that
        // above didn't find a match.
        let mut cur =
            start.add(USIZE_BYTES - (start.as_usize() & USIZE_ALIGN));
        debug_assert!(cur > start);
        if len <= One::LOOP_BYTES {
            return generic::fwd_byte_by_byte(cur, end, confirm);
        }
        debug_assert!(end.sub(One::LOOP_BYTES) >= start);
        while cur <= end.sub(One::LOOP_BYTES) {
            debug_assert_eq!(0, cur.as_usize() % USIZE_BYTES);

            let a = cur.cast::<usize>().read();
            let b = cur.add(USIZE_BYTES).cast::<usize>().read();
            if self.has_needle(a) || self.has_needle(b) {
                break;
            }
            cur = cur.add(One::LOOP_BYTES);
        }
        generic::fwd_byte_by_byte(cur, end, confirm)
    }

    /// Like `rfind`, but accepts and returns raw pointers.
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
    pub unsafe fn rfind_raw(
        &self,
        start: *const u8,
        end: *const u8,
    ) -> Option<*const u8> {
        if start >= end {
            return None;
        }
        let confirm = |b| self.confirm(b);
        let len = end.distance(start);
        if len < USIZE_BYTES {
            return generic::rev_byte_by_byte(start, end, confirm);
        }

        let chunk = end.sub(USIZE_BYTES).cast::<usize>().read_unaligned();
        if self.has_needle(chunk) {
            return generic::rev_byte_by_byte(start, end, confirm);
        }

        let mut cur = end.sub(end.as_usize() & USIZE_ALIGN);
        debug_assert!(start <= cur && cur <= end);
        if len <= One::LOOP_BYTES {
            return generic::rev_byte_by_byte(start, cur, confirm);
        }
        while cur >= start.add(One::LOOP_BYTES) {
            debug_assert_eq!(0, cur.as_usize() % USIZE_BYTES);

            let a = cur.sub(2 * USIZE_BYTES).cast::<usize>().read();
            let b = cur.sub(1 * USIZE_BYTES).cast::<usize>().read();
            if self.has_needle(a) || self.has_needle(b) {
                break;
            }
            cur = cur.sub(One::LOOP_BYTES);
        }
        generic::rev_byte_by_byte(start, cur, confirm)
    }

    /// Returns an iterator over all occurrences of the needle byte in the
    /// given haystack.
    ///
    /// The iterator returned implements `DoubleEndedIterator`. This means it
    /// can also be used to find occurrences in reverse order.
    pub fn iter<'a, 'h>(&'a self, haystack: &'h [u8]) -> OneIter<'a, 'h> {
        OneIter { searcher: self, it: generic::Iter::new(haystack) }
    }

    #[inline(always)]
    fn has_needle(&self, chunk: usize) -> bool {
        has_zero_byte(self.v1 ^ chunk)
    }

    #[inline(always)]
    fn confirm(&self, haystack_byte: u8) -> bool {
        self.s1 == haystack_byte
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
    /// The underlying memchr searcher.
    searcher: &'a One,
    /// Generic iterator implementation.
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
        unsafe { self.it.next_back(|s, e| self.searcher.rfind_raw(s, e)) }
    }
}

/// Finds all occurrences of two bytes in a haystack.
///
/// That is, this reports matches of one of two possible bytes. For example,
/// searching for `a` or `b` in `afoobar` would report matches at offsets `0`,
/// `4` and `5`.
#[derive(Clone, Debug)]
pub struct Two {
    s1: u8,
    s2: u8,
    v1: usize,
    v2: usize,
}

impl Two {
    /// Create a new searcher that finds occurrences of the two needle bytes
    /// given.
    #[inline]
    pub fn new(needle1: u8, needle2: u8) -> Two {
        Two {
            s1: needle1,
            s2: needle2,
            v1: splat(needle1),
            v2: splat(needle2),
        }
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

    /// Return the last occurrence of one of the needle bytes in the given
    /// haystack. If no such occurrence exists, then `None` is returned.
    ///
    /// The occurrence is reported as an offset into `haystack`. Its maximum
    /// value for a non-empty haystack is `haystack.len() - 1`.
    #[inline]
    pub fn rfind(&self, haystack: &[u8]) -> Option<usize> {
        // SAFETY: `find_raw` guarantees that if a pointer is returned, it
        // falls within the bounds of the start and end pointers.
        unsafe {
            generic::search_slice_with_raw(haystack, |s, e| {
                self.rfind_raw(s, e)
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
        if start >= end {
            return None;
        }
        let confirm = |b| self.confirm(b);
        let len = end.distance(start);
        if len < USIZE_BYTES {
            return generic::fwd_byte_by_byte(start, end, confirm);
        }

        // The start of the search may not be aligned to `*const usize`,
        // so we do an unaligned load here.
        let chunk = start.cast::<usize>().read_unaligned();
        if self.has_needle(chunk) {
            return generic::fwd_byte_by_byte(start, end, confirm);
        }

        // And now we start our search at a guaranteed aligned position.
        // The first iteration of the loop below will overlap with the the
        // unaligned chunk above in cases where the search starts at an
        // unaligned offset, but that's okay as we're only here if that
        // above didn't find a match.
        let mut cur =
            start.add(USIZE_BYTES - (start.as_usize() & USIZE_ALIGN));
        debug_assert!(cur > start);
        debug_assert!(end.sub(USIZE_BYTES) >= start);
        while cur <= end.sub(USIZE_BYTES) {
            debug_assert_eq!(0, cur.as_usize() % USIZE_BYTES);

            let chunk = cur.cast::<usize>().read();
            if self.has_needle(chunk) {
                break;
            }
            cur = cur.add(USIZE_BYTES);
        }
        generic::fwd_byte_by_byte(cur, end, confirm)
    }

    /// Like `rfind`, but accepts and returns raw pointers.
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
    pub unsafe fn rfind_raw(
        &self,
        start: *const u8,
        end: *const u8,
    ) -> Option<*const u8> {
        if start >= end {
            return None;
        }
        let confirm = |b| self.confirm(b);
        let len = end.distance(start);
        if len < USIZE_BYTES {
            return generic::rev_byte_by_byte(start, end, confirm);
        }

        let chunk = end.sub(USIZE_BYTES).cast::<usize>().read_unaligned();
        if self.has_needle(chunk) {
            return generic::rev_byte_by_byte(start, end, confirm);
        }

        let mut cur = end.sub(end.as_usize() & USIZE_ALIGN);
        debug_assert!(start <= cur && cur <= end);
        while cur >= start.add(USIZE_BYTES) {
            debug_assert_eq!(0, cur.as_usize() % USIZE_BYTES);

            let chunk = cur.sub(USIZE_BYTES).cast::<usize>().read();
            if self.has_needle(chunk) {
                break;
            }
            cur = cur.sub(USIZE_BYTES);
        }
        generic::rev_byte_by_byte(start, cur, confirm)
    }

    /// Returns an iterator over all occurrences of one of the needle bytes in
    /// the given haystack.
    ///
    /// The iterator returned implements `DoubleEndedIterator`. This means it
    /// can also be used to find occurrences in reverse order.
    pub fn iter<'a, 'h>(&'a self, haystack: &'h [u8]) -> TwoIter<'a, 'h> {
        TwoIter { searcher: self, it: generic::Iter::new(haystack) }
    }

    #[inline(always)]
    fn has_needle(&self, chunk: usize) -> bool {
        has_zero_byte(self.v1 ^ chunk) || has_zero_byte(self.v2 ^ chunk)
    }

    #[inline(always)]
    fn confirm(&self, haystack_byte: u8) -> bool {
        self.s1 == haystack_byte || self.s2 == haystack_byte
    }
}

/// An iterator over all occurrences of two possible bytes in a haystack.
///
/// This iterator implements `DoubleEndedIterator`, which means it can also be
/// used to find occurrences in reverse order.
///
/// This iterator is created by the [`Two::iter`] method.
///
/// The lifetime parameters are as follows:
///
/// * `'a` refers to the lifetime of the underlying [`Two`] searcher.
/// * `'h` refers to the lifetime of the haystack being searched.
#[derive(Clone, Debug)]
pub struct TwoIter<'a, 'h> {
    /// The underlying memchr searcher.
    searcher: &'a Two,
    /// Generic iterator implementation.
    it: generic::Iter<'h>,
}

impl<'a, 'h> Iterator for TwoIter<'a, 'h> {
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

impl<'a, 'h> DoubleEndedIterator for TwoIter<'a, 'h> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        // SAFETY: We rely on the generic iterator to provide valid start
        // and end pointers, but we guarantee that any pointer returned by
        // 'rfind_raw' falls within the bounds of the start and end pointer.
        unsafe { self.it.next_back(|s, e| self.searcher.rfind_raw(s, e)) }
    }
}

/// Finds all occurrences of three bytes in a haystack.
///
/// That is, this reports matches of one of three possible bytes. For example,
/// searching for `a`, `b` or `o` in `afoobar` would report matches at offsets
/// `0`, `2`, `3`, `4` and `5`.
#[derive(Clone, Debug)]
pub struct Three {
    s1: u8,
    s2: u8,
    s3: u8,
    v1: usize,
    v2: usize,
    v3: usize,
}

impl Three {
    /// Create a new searcher that finds occurrences of the three needle bytes
    /// given.
    #[inline]
    pub fn new(needle1: u8, needle2: u8, needle3: u8) -> Three {
        Three {
            s1: needle1,
            s2: needle2,
            s3: needle3,
            v1: splat(needle1),
            v2: splat(needle2),
            v3: splat(needle3),
        }
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

    /// Return the last occurrence of one of the needle bytes in the given
    /// haystack. If no such occurrence exists, then `None` is returned.
    ///
    /// The occurrence is reported as an offset into `haystack`. Its maximum
    /// value for a non-empty haystack is `haystack.len() - 1`.
    #[inline]
    pub fn rfind(&self, haystack: &[u8]) -> Option<usize> {
        // SAFETY: `find_raw` guarantees that if a pointer is returned, it
        // falls within the bounds of the start and end pointers.
        unsafe {
            generic::search_slice_with_raw(haystack, |s, e| {
                self.rfind_raw(s, e)
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
        if start >= end {
            return None;
        }
        let confirm = |b| self.confirm(b);
        let len = end.distance(start);
        if len < USIZE_BYTES {
            return generic::fwd_byte_by_byte(start, end, confirm);
        }

        // The start of the search may not be aligned to `*const usize`,
        // so we do an unaligned load here.
        let chunk = start.cast::<usize>().read_unaligned();
        if self.has_needle(chunk) {
            return generic::fwd_byte_by_byte(start, end, confirm);
        }

        // And now we start our search at a guaranteed aligned position.
        // The first iteration of the loop below will overlap with the the
        // unaligned chunk above in cases where the search starts at an
        // unaligned offset, but that's okay as we're only here if that
        // above didn't find a match.
        let mut cur =
            start.add(USIZE_BYTES - (start.as_usize() & USIZE_ALIGN));
        debug_assert!(cur > start);
        debug_assert!(end.sub(USIZE_BYTES) >= start);
        while cur <= end.sub(USIZE_BYTES) {
            debug_assert_eq!(0, cur.as_usize() % USIZE_BYTES);

            let chunk = cur.cast::<usize>().read();
            if self.has_needle(chunk) {
                break;
            }
            cur = cur.add(USIZE_BYTES);
        }
        generic::fwd_byte_by_byte(cur, end, confirm)
    }

    /// Like `rfind`, but accepts and returns raw pointers.
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
    pub unsafe fn rfind_raw(
        &self,
        start: *const u8,
        end: *const u8,
    ) -> Option<*const u8> {
        if start >= end {
            return None;
        }
        let confirm = |b| self.confirm(b);
        let len = end.distance(start);
        if len < USIZE_BYTES {
            return generic::rev_byte_by_byte(start, end, confirm);
        }

        let chunk = end.sub(USIZE_BYTES).cast::<usize>().read_unaligned();
        if self.has_needle(chunk) {
            return generic::rev_byte_by_byte(start, end, confirm);
        }

        let mut cur = end.sub(end.as_usize() & USIZE_ALIGN);
        debug_assert!(start <= cur && cur <= end);
        while cur >= start.add(USIZE_BYTES) {
            debug_assert_eq!(0, cur.as_usize() % USIZE_BYTES);

            let chunk = cur.sub(USIZE_BYTES).cast::<usize>().read();
            if self.has_needle(chunk) {
                break;
            }
            cur = cur.sub(USIZE_BYTES);
        }
        generic::rev_byte_by_byte(start, cur, confirm)
    }

    /// Returns an iterator over all occurrences of one of the needle bytes in
    /// the given haystack.
    ///
    /// The iterator returned implements `DoubleEndedIterator`. This means it
    /// can also be used to find occurrences in reverse order.
    pub fn iter<'a, 'h>(&'a self, haystack: &'h [u8]) -> ThreeIter<'a, 'h> {
        ThreeIter { searcher: self, it: generic::Iter::new(haystack) }
    }

    #[inline(always)]
    fn has_needle(&self, chunk: usize) -> bool {
        has_zero_byte(self.v1 ^ chunk)
            || has_zero_byte(self.v2 ^ chunk)
            || has_zero_byte(self.v3 ^ chunk)
    }

    #[inline(always)]
    fn confirm(&self, haystack_byte: u8) -> bool {
        self.s1 == haystack_byte
            || self.s2 == haystack_byte
            || self.s3 == haystack_byte
    }
}

/// An iterator over all occurrences of three possible bytes in a haystack.
///
/// This iterator implements `DoubleEndedIterator`, which means it can also be
/// used to find occurrences in reverse order.
///
/// This iterator is created by the [`Three::iter`] method.
///
/// The lifetime parameters are as follows:
///
/// * `'a` refers to the lifetime of the underlying [`Three`] searcher.
/// * `'h` refers to the lifetime of the haystack being searched.
#[derive(Clone, Debug)]
pub struct ThreeIter<'a, 'h> {
    /// The underlying memchr searcher.
    searcher: &'a Three,
    /// Generic iterator implementation.
    it: generic::Iter<'h>,
}

impl<'a, 'h> Iterator for ThreeIter<'a, 'h> {
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

impl<'a, 'h> DoubleEndedIterator for ThreeIter<'a, 'h> {
    #[inline]
    fn next_back(&mut self) -> Option<usize> {
        // SAFETY: We rely on the generic iterator to provide valid start
        // and end pointers, but we guarantee that any pointer returned by
        // 'rfind_raw' falls within the bounds of the start and end pointer.
        unsafe { self.it.next_back(|s, e| self.searcher.rfind_raw(s, e)) }
    }
}

/// Return `true` if `x` contains any zero byte.
///
/// That is, this routine treats `x` as a register of 8-bit lanes and returns
/// true when any of those lanes is `0`.
///
/// From "Matters Computational" by J. Arndt.
#[inline(always)]
fn has_zero_byte(x: usize) -> bool {
    // "The idea is to subtract one from each of the bytes and then look for
    // bytes where the borrow propagated all the way to the most significant
    // bit."
    const LO: usize = splat(0x01);
    const HI: usize = splat(0x80);

    (x.wrapping_sub(LO) & !x & HI) != 0
}

/// Repeat the given byte into a word size number. That is, every 8 bits
/// is equivalent to the given byte. For example, if `b` is `\x4E` or
/// `01001110` in binary, then the returned value on a 32-bit system would be:
/// `01001110_01001110_01001110_01001110`.
#[inline(always)]
const fn splat(b: u8) -> usize {
    // TODO: use `usize::from` once it can be used in const context.
    (b as usize) * (usize::MAX / 255)
}

#[cfg(test)]
mod tests {
    use quickcheck::quickcheck;

    use crate::tests::memchr::{naive, testdata::memchr_tests};

    use super::*;

    #[test]
    fn memchr1() {
        for test in memchr_tests() {
            test.one(false, |n1, hay| One::new(n1).find(hay));
        }
    }

    #[test]
    fn memchr1_iter() {
        for test in memchr_tests() {
            test.all_one(false, |n1, hay| One::new(n1).iter(hay).collect());
        }
    }

    #[test]
    fn memrchr1() {
        for test in memchr_tests() {
            test.one(true, |n1, hay| One::new(n1).rfind(hay));
        }
    }

    #[test]
    fn memrchr1_iter() {
        for test in memchr_tests() {
            test.all_one(true, |n1, hay| {
                One::new(n1).iter(hay).rev().collect()
            });
        }
    }

    #[test]
    fn memchr2() {
        for test in memchr_tests() {
            test.two(false, |n1, n2, hay| Two::new(n1, n2).find(hay));
        }
    }

    #[test]
    fn memchr2_iter() {
        for test in memchr_tests() {
            test.all_two(false, |n1, n2, hay| {
                Two::new(n1, n2).iter(hay).collect()
            });
        }
    }

    #[test]
    fn memrchr2() {
        for test in memchr_tests() {
            test.two(true, |n1, n2, hay| Two::new(n1, n2).rfind(hay));
        }
    }

    #[test]
    fn memrchr2_iter() {
        for test in memchr_tests() {
            test.all_two(true, |n1, n2, hay| {
                Two::new(n1, n2).iter(hay).rev().collect()
            });
        }
    }

    #[test]
    fn memchr3() {
        for test in memchr_tests() {
            test.three(false, |n1, n2, n3, hay| {
                Three::new(n1, n2, n3).find(hay)
            });
        }
    }

    #[test]
    fn memchr3_iter() {
        for test in memchr_tests() {
            test.all_three(false, |n1, n2, n3, hay| {
                Three::new(n1, n2, n3).iter(hay).collect()
            });
        }
    }

    #[test]
    fn memrchr3() {
        for test in memchr_tests() {
            test.three(true, |n1, n2, n3, hay| {
                Three::new(n1, n2, n3).rfind(hay)
            });
        }
    }

    #[test]
    fn memrchr3_iter() {
        for test in memchr_tests() {
            test.all_three(true, |n1, n2, n3, hay| {
                Three::new(n1, n2, n3).iter(hay).rev().collect()
            });
        }
    }

    quickcheck! {
        fn qc_memchr1_matches_naive(n1: u8, corpus: Vec<u8>) -> bool {
            One::new(n1).find(&corpus) == naive::memchr(n1, &corpus)
        }
    }

    quickcheck! {
        fn qc_memchr2_matches_naive(n1: u8, n2: u8, corpus: Vec<u8>) -> bool {
            Two::new(n1, n2).find(&corpus) == naive::memchr2(n1, n2, &corpus)
        }
    }

    quickcheck! {
        fn qc_memchr3_matches_naive(
            n1: u8, n2: u8, n3: u8,
            corpus: Vec<u8>
        ) -> bool {
            Three::new(n1, n2, n3).find(&corpus)
                == naive::memchr3(n1, n2, n3, &corpus)
        }
    }

    quickcheck! {
        fn qc_memrchr1_matches_naive(n1: u8, corpus: Vec<u8>) -> bool {
            One::new(n1).rfind(&corpus) == naive::memrchr(n1, &corpus)
        }
    }

    quickcheck! {
        fn qc_memrchr2_matches_naive(n1: u8, n2: u8, corpus: Vec<u8>) -> bool {
            Two::new(n1, n2).rfind(&corpus) == naive::memrchr2(n1, n2, &corpus)
        }
    }

    quickcheck! {
        fn qc_memrchr3_matches_naive(
            n1: u8, n2: u8, n3: u8,
            corpus: Vec<u8>
        ) -> bool {
            Three::new(n1, n2, n3).rfind(&corpus)
                == naive::memrchr3(n1, n2, n3, &corpus)
        }
    }
}
