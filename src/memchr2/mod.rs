use core::iter::Rev;

pub(crate) mod naive;
mod searcher;

#[inline]
pub fn memchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    naive::memchr(needle, haystack)
}

#[inline]
pub fn memchr2(needle1: u8, needle2: u8, haystack: &[u8]) -> Option<usize> {
    naive::memchr2(needle1, needle2, haystack)
}

#[inline]
pub fn memchr3(
    needle1: u8,
    needle2: u8,
    needle3: u8,
    haystack: &[u8],
) -> Option<usize> {
    naive::memchr3(needle1, needle2, needle3, haystack)
}

#[inline]
pub fn memrchr(needle: u8, haystack: &[u8]) -> Option<usize> {
    naive::memrchr(needle, haystack)
}

#[inline]
pub fn memrchr2(needle1: u8, needle2: u8, haystack: &[u8]) -> Option<usize> {
    naive::memrchr2(needle1, needle2, haystack)
}

#[inline]
pub fn memrchr3(
    needle1: u8,
    needle2: u8,
    needle3: u8,
    haystack: &[u8],
) -> Option<usize> {
    naive::memrchr3(needle1, needle2, needle3, haystack)
}

/// An iterator over all occurrences of the needle in a haystack.
#[inline]
pub fn memchr_iter(needle: u8, haystack: &[u8]) -> Memchr<'_> {
    Memchr::new(needle, haystack)
}

/// An iterator over all occurrences of the needles in a haystack.
#[inline]
pub fn memchr2_iter(needle1: u8, needle2: u8, haystack: &[u8]) -> Memchr2<'_> {
    Memchr2::new(needle1, needle2, haystack)
}

/// An iterator over all occurrences of the needles in a haystack.
#[inline]
pub fn memchr3_iter(
    needle1: u8,
    needle2: u8,
    needle3: u8,
    haystack: &[u8],
) -> Memchr3<'_> {
    Memchr3::new(needle1, needle2, needle3, haystack)
}

/// An iterator over all occurrences of the needle in a haystack, in reverse.
#[inline]
pub fn memrchr_iter(needle: u8, haystack: &[u8]) -> Rev<Memchr<'_>> {
    Memchr::new(needle, haystack).rev()
}

/// An iterator over all occurrences of the needles in a haystack, in reverse.
#[inline]
pub fn memrchr2_iter(
    needle1: u8,
    needle2: u8,
    haystack: &[u8],
) -> Rev<Memchr2<'_>> {
    Memchr2::new(needle1, needle2, haystack).rev()
}

/// An iterator over all occurrences of the needles in a haystack, in reverse.
#[inline]
pub fn memrchr3_iter(
    needle1: u8,
    needle2: u8,
    needle3: u8,
    haystack: &[u8],
) -> Rev<Memchr3<'_>> {
    Memchr3::new(needle1, needle2, needle3, haystack).rev()
}

/// An iterator for `memchr`.
pub struct Memchr<'a> {
    needle: u8,
    haystack: &'a [u8],
    fwd: usize,
    rev: usize,
}

impl<'a> Memchr<'a> {
    /// Creates a new iterator that yields all positions of needle in haystack.
    #[inline]
    pub fn new(needle: u8, haystack: &[u8]) -> Memchr<'_> {
        Memchr { needle, haystack, fwd: 0, rev: haystack.len() }
    }
}

impl<'a> Iterator for Memchr<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let result = self.fwd
            + memchr(self.needle, &self.haystack[self.fwd..self.rev])?;
        self.fwd = result + 1;
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.rev.saturating_sub(self.fwd)))
    }
}

impl<'a> DoubleEndedIterator for Memchr<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let result = self.fwd
            + memrchr(self.needle, &self.haystack[self.fwd..self.rev])?;
        self.rev = result;
        Some(result)
    }
}

/// An iterator for `memchr2`.
pub struct Memchr2<'a> {
    needle1: u8,
    needle2: u8,
    haystack: &'a [u8],
    fwd: usize,
    rev: usize,
}

impl<'a> Memchr2<'a> {
    /// Creates a new iterator that yields all positions of needle in haystack.
    #[inline]
    pub fn new(needle1: u8, needle2: u8, haystack: &[u8]) -> Memchr2<'_> {
        Memchr2 { needle1, needle2, haystack, fwd: 0, rev: haystack.len() }
    }
}

impl<'a> Iterator for Memchr2<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let result = self.fwd
            + memchr2(
                self.needle1,
                self.needle2,
                &self.haystack[self.fwd..self.rev],
            )?;
        self.fwd = result + 1;
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.rev.saturating_sub(self.fwd)))
    }
}

impl<'a> DoubleEndedIterator for Memchr2<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let result = self.fwd
            + memrchr2(
                self.needle1,
                self.needle2,
                &self.haystack[self.fwd..self.rev],
            )?;
        self.rev = result;
        Some(result)
    }
}

/// An iterator for `memchr3`.
pub struct Memchr3<'a> {
    needle1: u8,
    needle2: u8,
    needle3: u8,
    haystack: &'a [u8],
    fwd: usize,
    rev: usize,
}

impl<'a> Memchr3<'a> {
    /// Create a new `Memchr3` that's initialized to zero with a haystack
    #[inline]
    pub fn new(
        needle1: u8,
        needle2: u8,
        needle3: u8,
        haystack: &[u8],
    ) -> Memchr3<'_> {
        Memchr3 {
            needle1,
            needle2,
            needle3,
            haystack,
            fwd: 0,
            rev: haystack.len(),
        }
    }
}

impl<'a> Iterator for Memchr3<'a> {
    type Item = usize;

    #[inline]
    fn next(&mut self) -> Option<usize> {
        let result = self.fwd
            + memchr3(
                self.needle1,
                self.needle2,
                self.needle3,
                &self.haystack[self.fwd..self.rev],
            )?;
        self.fwd = result + 1;
        Some(result)
    }

    #[inline]
    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.rev.saturating_sub(self.fwd)))
    }
}

impl<'a> DoubleEndedIterator for Memchr3<'a> {
    #[inline]
    fn next_back(&mut self) -> Option<Self::Item> {
        let result = self.fwd
            + memrchr3(
                self.needle1,
                self.needle2,
                self.needle3,
                &self.haystack[self.fwd..self.rev],
            )?;
        self.rev = result;
        Some(result)
    }
}
