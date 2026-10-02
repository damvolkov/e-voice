use std::collections::VecDeque;
use std::time::Duration;

use crate::schema::emotion::Emotion;
use crate::schema::error::NodeError;
use crate::schema::segment::{SegmentId, SegmentSpan};

/// Lifecycle of one node's result for one segment: nodes resolve only what was requested, the first settlement wins.
#[derive(Debug, Clone, PartialEq)]
pub enum Branch<T> {
    Waiting,
    Running { deadline: Duration },
    Ready(Result<T, NodeError>),
}

impl<T> Branch<T> {
    #[must_use]
    pub const fn ready(&self) -> bool {
        matches!(self, Self::Ready(_))
    }

    pub fn settle(&mut self, result: Result<T, NodeError>) {
        match self {
            Self::Ready(_) => {}
            Self::Waiting | Self::Running { .. } => *self = Self::Ready(result),
        }
    }

    pub fn accept(&mut self, result: Result<T, NodeError>) {
        match self {
            Self::Running { .. } => *self = Self::Ready(result),
            Self::Waiting | Self::Ready(_) => {}
        }
    }

    pub fn expire(&mut self, now: Duration) -> bool {
        match self {
            Self::Running { deadline } if *deadline <= now => {
                *self = Self::Ready(Err(NodeError::Timeout));
                true
            }
            Self::Waiting | Self::Running { .. } | Self::Ready(_) => false,
        }
    }
}

/// Join state of one segment across every node that contributes to its final event.
#[derive(Debug, Clone, PartialEq)]
pub struct Slot {
    pub span: SegmentSpan,
    pub open: bool,
    pub opened: Duration,
    pub dropped: bool,
    pub asr: Branch<String>,
    pub ser: Branch<Emotion>,
}

impl Slot {
    #[must_use]
    pub const fn resolved(&self) -> bool {
        self.asr.ready() && self.ser.ready()
    }

    #[must_use]
    pub const fn done(&self) -> bool {
        !self.open && self.resolved()
    }

    #[must_use]
    pub const fn live(&self) -> bool {
        !self.dropped && !self.resolved()
    }

    pub fn fail(&mut self, error: &NodeError) {
        self.open = false;
        self.asr.settle(Err(error.clone()));
        self.ser.settle(Err(error.clone()));
    }
}

/// Pending segments in id order; released strictly from the front, so finals leave in order.
#[derive(Debug, Clone, Default)]
pub struct Join {
    slots: VecDeque<Slot>,
    front: u64,
    next: u64,
}

impl Join {
    #[must_use]
    pub fn live(&self) -> usize {
        self.slots.iter().filter(|slot| slot.live()).count()
    }

    /// Live segments whose speech already ended: work queued or running in nodes.
    #[must_use]
    pub fn backlog(&self) -> usize {
        self.slots.iter().filter(|slot| slot.live() && !slot.open).count()
    }

    #[must_use]
    pub fn is_empty(&self) -> bool {
        self.slots.is_empty()
    }

    pub fn open(&mut self, start: u64, now: Duration, dropped: bool) -> SegmentId {
        let id = SegmentId(self.next);
        self.next = self.next.saturating_add(1);
        self.slots.push_back(Slot {
            span: SegmentSpan { start, end: start },
            open: true,
            opened: now,
            dropped,
            asr: Branch::Waiting,
            ser: Branch::Waiting,
        });
        id
    }

    pub fn slot(&mut self, id: SegmentId) -> Option<&mut Slot> {
        id.0.checked_sub(self.front)
            .and_then(|offset| usize::try_from(offset).ok())
            .and_then(|index| self.slots.get_mut(index))
    }

    pub fn slots(&mut self) -> impl Iterator<Item = (SegmentId, &mut Slot)> {
        (self.front..).map(SegmentId).zip(self.slots.iter_mut())
    }

    #[must_use]
    pub fn oldest(&self) -> Option<SegmentId> {
        (self.front..)
            .zip(&self.slots)
            .find(|(_, slot)| slot.live())
            .map(|(id, _)| SegmentId(id))
    }

    pub fn release(&mut self) -> Option<(SegmentId, Slot)> {
        let id = SegmentId(self.front);
        let slot = self.slots.pop_front_if(|slot| slot.done())?;
        self.front = self.front.saturating_add(1);
        Some((id, slot))
    }
}
