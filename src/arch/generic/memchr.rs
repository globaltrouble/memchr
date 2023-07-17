/*!
A generic crate-internal routine for the `memchr` family of functions.
*/

// What follows is a vector algorithm generic over the specific vector
// type to detect the position of one, two or three needles in a
// haystack. From what I know, this is a "classic" algorithm. I believe
// it can be found in places like glibc and Go's standard library. It
// appears to be well known and is elaborated on in more detail here:
// https://gms.tf/stdfind-and-memchr-optimizations.html
//
// While the routine below is fairly long and perhaps intimidating, the basic
// idea is actually very simple and can be expressed straight-forwardly in
// pseudo code. The psuedo code below is written for 128 bit vectors, but the
// actual code below works for anything that implements the Vector trait.
//
//     needle = (n1 << 15) | (n1 << 14) | ... | (n1 << 1) | n1
//     // Note: shift amount is in bytes
//
//     while i <= haystack.len() - 16:
//       // A 16 byte vector. Each byte in chunk corresponds to a byte in
//       // the haystack.
//       chunk = haystack[i:i+16]
//       // Compare bytes in needle with bytes in chunk. The result is a 16
//       // byte chunk where each byte is 0xFF if the corresponding bytes
//       // in needle and chunk were equal, or 0x00 otherwise.
//       eqs = cmpeq(needle, chunk)
//       // Return a 32 bit integer where the most significant 16 bits
//       // are always 0 and the lower 16 bits correspond to whether the
//       // most significant bit in the correspond byte in `eqs` is set.
//       // In other words, `mask as u16` has bit i set if and only if
//       // needle[i] == chunk[i].
//       mask = movemask(eqs)
//
//       // Mask is 0 if there is no match, and non-zero otherwise.
//       if mask != 0:
//         // trailing_zeros tells us the position of the least significant
//         // bit that is set.
//         return i + trailing_zeros(mask)
//
//     // haystack length may not be a multiple of 16, so search the rest.
//     while i < haystack.len():
//       if haystack[i] == n1:
//         return i
//
//     // No match found.
//     return NULL
//
// In fact, we could loosely translate the above code to Rust line-for-line
// and it would be a pretty fast algorithm. But, we pull out all the stops
// to go as fast as possible:
//
// 1. We use aligned loads. That is, we do some finagling to make sure our
//    primary loop not only proceeds in increments of 16 bytes, but that
//    the address of haystack's pointer that we dereference is aligned to
//    16 bytes. 16 is a magic number here because it is the size of SSE2
//    128-bit vector. (For the AVX2 algorithm, 32 is the magic number.)
//    Therefore, to get aligned loads, our pointer's address must be evenly
//    divisible by 16.
// 2. Our primary loop proceeds 64 bytes at a time instead of 16. It's
//    kind of like loop unrolling, but we combine the equality comparisons
//    using a vector OR such that we only need to extract a single mask to
//    determine whether a match exists or not. If so, then we do some
//    book-keeping to determine the precise location but otherwise mush on.
// 3. We use our "chunk" comparison routine in as many places as possible,
//    even if it means using unaligned loads. In particular, if haystack
//    starts with an unaligned address, then we do an unaligned load to
//    search the first 16 bytes. We then start our primary loop at the
//    smallest subsequent aligned address, which will actually overlap with
//    previously searched bytes. But we're OK with that. We do a similar
//    dance at the end of our primary loop. Finally, to avoid a
//    byte-at-a-time loop at the end, we do a final 16 byte unaligned load
//    that may overlap with a previous load. This is OK because it converts
//    a loop into a small number of very fast vector instructions.
//
// In general, most of the algorithms in this crate have a similar structure to
// what you see below, so this comment applies fairly well to all of them.

use crate::{ext::Pointer, vector::Vector};

#[derive(Clone, Debug)]
pub(crate) struct One<V> {
    s1: u8,
    v1: V,
}

impl<V: Vector> One<V> {
    const LOOP_SIZE: usize = 4 * V::BYTES;
    const VECTOR_ALIGN: usize = V::BYTES - 1;

    #[inline(always)]
    pub(crate) unsafe fn new(needle: u8) -> One<V> {
        One { s1: needle, v1: V::splat(needle) }
    }

    #[inline(always)]
    pub(crate) fn needle1(&self) -> u8 {
        self.s1
    }

    #[inline(always)]
    pub(crate) unsafe fn find_raw(
        &self,
        start: *const u8,
        end: *const u8,
    ) -> Option<*const u8> {
        let len = end.distance(start);
        let loop_size = core::cmp::min(One::<V>::LOOP_SIZE, len);
        let mut cur = start;

        if let Some(i) = self.forward_search1(start, end, cur) {
            return Some(i);
        }

        cur = cur.add(V::BYTES - (start as usize & One::<V>::VECTOR_ALIGN));
        debug_assert!(cur > start && end.sub(V::BYTES) >= start);
        while loop_size == One::<V>::LOOP_SIZE && cur <= end.sub(loop_size) {
            debug_assert_eq!(0, (cur as usize) % V::BYTES);

            let a = V::load_aligned(cur);
            let b = V::load_aligned(cur.add(1 * V::BYTES));
            let c = V::load_aligned(cur.add(2 * V::BYTES));
            let d = V::load_aligned(cur.add(3 * V::BYTES));
            let eqa = self.v1.cmpeq(a);
            let eqb = self.v1.cmpeq(b);
            let eqc = self.v1.cmpeq(c);
            let eqd = self.v1.cmpeq(d);
            let or1 = eqa.or(eqb);
            let or2 = eqc.or(eqd);
            let or3 = or1.or(or2);
            if or3.movemask() != 0 {
                let mask = eqa.movemask();
                if mask != 0 {
                    return Some(cur.add(self.forward_pos(mask)));
                }

                cur = cur.add(V::BYTES);
                let mask = eqb.movemask();
                if mask != 0 {
                    return Some(cur.add(self.forward_pos(mask)));
                }

                cur = cur.add(V::BYTES);
                let mask = eqc.movemask();
                if mask != 0 {
                    return Some(cur.add(self.forward_pos(mask)));
                }

                cur = cur.add(V::BYTES);
                let mask = eqd.movemask();
                debug_assert!(mask != 0);
                return Some(cur.add(self.forward_pos(mask)));
            }
            cur = cur.add(loop_size);
        }
        while cur <= end.sub(V::BYTES) {
            debug_assert!(end.distance(cur) >= V::BYTES);

            if let Some(i) = self.forward_search1(start, end, cur) {
                return Some(i);
            }
            cur = cur.add(V::BYTES);
        }
        if cur < end {
            debug_assert!(end.distance(cur) < V::BYTES);
            cur = cur.sub(V::BYTES - end.distance(cur));
            debug_assert_eq!(end.distance(cur), V::BYTES);

            return self.forward_search1(start, end, cur);
        }
        None
    }

    #[inline(always)]
    pub(crate) unsafe fn rfind_raw(
        &self,
        start: *const u8,
        end: *const u8,
    ) -> Option<*const u8> {
        todo!()
    }

    #[inline(always)]
    unsafe fn forward_search1(
        &self,
        start: *const u8,
        end: *const u8,
        cur: *const u8,
    ) -> Option<*const u8> {
        debug_assert!(end.distance(start) >= V::BYTES);
        debug_assert!(start <= cur);
        debug_assert!(cur <= end.sub(V::BYTES));

        let chunk = V::load_unaligned(cur);
        let mask = self.v1.cmpeq(chunk).movemask();
        if mask != 0 {
            Some(cur.add(self.forward_pos(mask)))
        } else {
            None
        }
    }

    /// Compute the position of the first matching byte from the given mask. The
    /// position returned is always in the range [0, 15].
    ///
    /// The mask given is expected to be the result of _mm_movemask_epi8.
    #[inline(always)]
    fn forward_pos(&self, mask: u32) -> usize {
        // We are dealing with little endian here, where the most significant
        // byte is at a higher address. That means the least significant bit
        // that is set corresponds to the position of our first matching byte.
        // That position corresponds to the number of zeros after the least
        // significant bit.
        mask.trailing_zeros() as usize
    }
}

/// An iterator over all occurrences of a set of bytes in a haystack.
///
/// This iterator implements the routines necessary to provide a
/// `DoubleEndedIterator` impl, which means it can also be used to find
/// occurrences in reverse order.
///
/// The lifetime parameters are as follows:
///
/// * `'h` refers to the lifetime of the haystack being searched.
///
/// This type is intended to be used to implement all iterators for the
/// `memchr` family of functions. It handles a tiny bit of marginally tricky
/// raw pointer math, but otherwise expects the caller to provide `find_raw`
/// and `rfind_raw` routines for each call of `next` and `next_back`,
/// respectively.
#[derive(Clone, Debug)]
pub(crate) struct Iter<'h> {
    /// The original starting point into the haystack. We use this to convert
    /// pointers to offsets.
    original_start: *const u8,
    /// The current starting point into the haystack. That is, where the next
    /// search will begin.
    start: *const u8,
    /// The current ending point into the haystack. That is, where the next
    /// reverse search will begin.
    end: *const u8,
    /// A marker for tracking the lifetime of the start/cur_start/cur_end
    /// pointers above, which all point into the haystack.
    haystack: core::marker::PhantomData<&'h [u8]>,
}

impl<'h> Iter<'h> {
    /// Create a new generic memchr iterator.
    #[inline(always)]
    pub(crate) fn new(haystack: &'h [u8]) -> Iter<'h> {
        Iter {
            original_start: haystack.as_ptr(),
            start: haystack.as_ptr(),
            end: haystack.as_ptr().wrapping_add(haystack.len()),
            haystack: core::marker::PhantomData,
        }
    }

    /// Returns the next occurrence in the forward direction.
    ///
    /// # Safety
    ///
    /// Callers must ensure that if a pointer is returned from the closure
    /// provided, then it must be greater than or equal to the start pointer
    /// and less than the end pointer.
    #[inline(always)]
    pub(crate) unsafe fn next(
        &mut self,
        mut find_raw: impl FnMut(*const u8, *const u8) -> Option<*const u8>,
    ) -> Option<usize> {
        // SAFETY: Pointers are derived directly from the same &[u8] haystack.
        // We only ever modify start/end corresponding to a matching offset
        // found between start and end. Thus all changes to start/end maintain
        // our safety requirements.
        //
        // The only other assumption we rely on is that the pointer returned
        // by `find_raw` satisfies `self.start <= found < self.end`, and that
        // safety contract is forwarded to the caller.
        let found = find_raw(self.start, self.end)?;
        let result = found.distance(self.original_start);
        self.start = found.add(1);
        Some(result)
    }

    /// Returns the next occurrence in reverse.
    ///
    /// # Safety
    ///
    /// Callers must ensure that if a pointer is returned from the closure
    /// provided, then it must be greater than or equal to the start pointer
    /// and less than the end pointer.
    #[inline(always)]
    pub(crate) unsafe fn next_back(
        &mut self,
        mut rfind_raw: impl FnMut(*const u8, *const u8) -> Option<*const u8>,
    ) -> Option<usize> {
        // SAFETY: Pointers are derived directly from the same &[u8] haystack.
        // We only ever modify start/end corresponding to a matching offset
        // found between start and end. Thus all changes to start/end maintain
        // our safety requirements.
        //
        // The only other assumption we rely on is that the pointer returned
        // by `rfind_raw` satisfies `self.start <= found < self.end`, and that
        // safety contract is forwarded to the caller.
        let found = rfind_raw(self.original_start, self.end)?;
        let result = found.distance(self.original_start);
        self.end = found;
        Some(result)
    }

    /// Provides an implementation of `Iterator::size_hint`.
    #[inline(always)]
    pub(crate) fn size_hint(&self) -> (usize, Option<usize>) {
        (
            0,
            Some(
                self.end
                    .as_usize()
                    .saturating_sub(self.original_start.as_usize()),
            ),
        )
    }
}

/// Search a slice using a function that operates on raw pointers.
///
/// Given a function to search a contiguous sequence of memory for the location
/// of a non-empty set of bytes, this will execute that search on a slice of
/// bytes. The pointer returned by the given function will be converted to an
/// offset relative to the starting point of the given slice. That is, if a
/// match is found, the offset returned by this routine is guaranteed to be a
/// valid index into `haystack`.
///
/// Callers may use this for a forward or reverse search.
///
/// # Safety
///
/// Callers must ensure that if a pointer is returned by `find_raw`, then the
/// pointer must be greater than or equal to the starting pointer and less than
/// the end pointer.
#[inline(always)]
pub(crate) unsafe fn search_slice_with_raw(
    haystack: &[u8],
    mut find_raw: impl FnMut(*const u8, *const u8) -> Option<*const u8>,
) -> Option<usize> {
    // SAFETY: We rely on `find_raw` to return a correct and valid pointer, but
    // otherwise, `start` and `end` are valid due to the guarantees provided by
    // a &[u8].
    let start = haystack.as_ptr();
    let end = start.add(haystack.len());
    let found = find_raw(start, end)?;
    Some(found.distance(start))
}

/// Performs a forward byte-at-a-time loop until either `ptr >= end_ptr` or
/// until `confirm(*ptr)` returns `true`. If the former occurs, then `None` is
/// returned. If the latter occurs, then the pointer at which `confirm` returns
/// `true` is returned.
///
/// # Safety
///
/// Callers must provide valid pointers and they must satisfy `start_ptr <=
/// ptr` and `ptr <= end_ptr`.
#[inline(always)]
pub(crate) unsafe fn fwd_byte_by_byte<F: Fn(u8) -> bool>(
    start: *const u8,
    end: *const u8,
    confirm: F,
) -> Option<*const u8> {
    debug_assert!(start <= end);
    let mut ptr = start;
    while ptr < end {
        if confirm(*ptr) {
            return Some(ptr);
        }
        ptr = ptr.offset(1);
    }
    None
}

/// Performs a reverse byte-at-a-time loop until either `ptr < start_ptr` or
/// until `confirm(*ptr)` returns `true`. If the former occurs, then `None` is
/// returned. If the latter occurs, then the pointer at which `confirm` returns
/// `true` is returned.
///
/// # Safety
///
/// Callers must provide valid pointers and they must satisfy `start_ptr <=
/// ptr` and `ptr <= end_ptr`.
#[inline(always)]
pub(crate) unsafe fn rev_byte_by_byte<F: Fn(u8) -> bool>(
    start: *const u8,
    end: *const u8,
    confirm: F,
) -> Option<*const u8> {
    debug_assert!(start <= end);

    let mut ptr = end;
    while ptr > start {
        ptr = ptr.offset(-1);
        if confirm(*ptr) {
            return Some(ptr);
        }
    }
    None
}
