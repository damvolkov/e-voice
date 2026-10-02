use std::time::Duration;

/// When an opened gate falls back to listening for the wake word.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GatePolicy {
    Disabled,
    Utterance { idle: Duration },
    Window { idle: Duration },
    Session,
}

/// Where incoming audio must be sent.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateRoute {
    Wake,
    Vad,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum State {
    Listening,
    Open { last: Duration },
}

/// Wake-word gate: a pure state machine driven by detections, speech activity and time.
#[derive(Debug, Clone)]
pub struct Gate {
    policy: GatePolicy,
    state: State,
}

impl Gate {
    // ##### PRIVATE #####

    fn shift(&mut self, state: State) -> Option<GateRoute> {
        let before = self.route();
        self.state = state;
        let after = self.route();
        (before != after).then_some(after)
    }

    // ##########################################################

    // ##### PUBLIC #####

    #[must_use]
    pub const fn new(policy: GatePolicy) -> Self {
        let state = match policy {
            GatePolicy::Disabled => State::Open { last: Duration::ZERO },
            GatePolicy::Utterance { .. } | GatePolicy::Window { .. } | GatePolicy::Session => State::Listening,
        };
        Self { policy, state }
    }

    #[must_use]
    pub const fn enabled(&self) -> bool {
        !matches!(self.policy, GatePolicy::Disabled)
    }

    #[must_use]
    pub const fn route(&self) -> GateRoute {
        match self.state {
            State::Listening => GateRoute::Wake,
            State::Open { .. } => GateRoute::Vad,
        }
    }

    pub fn wake(&mut self, now: Duration) -> Option<GateRoute> {
        match self.policy {
            GatePolicy::Disabled => None,
            GatePolicy::Utterance { .. } | GatePolicy::Window { .. } | GatePolicy::Session => {
                self.shift(State::Open { last: now })
            }
        }
    }

    pub fn touch(&mut self, now: Duration) {
        match self.state {
            State::Open { .. } => self.state = State::Open { last: now },
            State::Listening => {}
        }
    }

    pub fn end(&mut self, now: Duration) -> Option<GateRoute> {
        match self.policy {
            GatePolicy::Utterance { .. } => self.shift(State::Listening),
            GatePolicy::Disabled | GatePolicy::Window { .. } | GatePolicy::Session => {
                self.touch(now);
                None
            }
        }
    }

    pub fn tick(&mut self, now: Duration, speaking: bool) -> Option<GateRoute> {
        match (self.policy, self.state, speaking) {
            (GatePolicy::Utterance { idle } | GatePolicy::Window { idle }, State::Open { last }, false)
                if now.saturating_sub(last) >= idle =>
            {
                self.shift(State::Listening)
            }
            _ => None,
        }
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use crate::workflow::gate::{Gate, GatePolicy, GateRoute};

    const IDLE: Duration = Duration::from_secs(5);

    fn secs(n: u64) -> Duration {
        Duration::from_secs(n)
    }

    #[test]
    fn test_disabled_always_routes_to_vad() {
        let mut gate = Gate::new(GatePolicy::Disabled);
        assert_eq!(gate.wake(secs(1)), None);
        assert_eq!(gate.end(secs(2)), None);
        assert_eq!(gate.tick(secs(100), false), None);
        assert_eq!(gate.route(), GateRoute::Vad);
    }

    #[test]
    fn test_wake_opens_once() {
        let mut gate = Gate::new(GatePolicy::Session);
        assert_eq!(gate.route(), GateRoute::Wake);
        assert_eq!(gate.wake(secs(1)), Some(GateRoute::Vad));
        assert_eq!(gate.wake(secs(2)), None);
    }

    #[test]
    fn test_utterance_closes_on_end() {
        let mut gate = Gate::new(GatePolicy::Utterance { idle: IDLE });
        gate.wake(secs(1));
        assert_eq!(gate.end(secs(2)), Some(GateRoute::Wake));
    }

    #[test]
    fn test_utterance_closes_when_idle_without_speech() {
        let mut gate = Gate::new(GatePolicy::Utterance { idle: IDLE });
        gate.wake(secs(1));
        assert_eq!(gate.tick(secs(6), false), Some(GateRoute::Wake));
    }

    #[test]
    fn test_window_survives_speech_and_closes_after_idle() {
        let mut gate = Gate::new(GatePolicy::Window { idle: IDLE });
        gate.wake(secs(1));
        assert_eq!(gate.tick(secs(10), true), None);
        gate.touch(secs(10));
        assert_eq!(gate.end(secs(12)), None);
        assert_eq!(gate.tick(secs(16), false), None);
        assert_eq!(gate.tick(secs(17), false), Some(GateRoute::Wake));
    }

    #[test]
    fn test_session_never_closes() {
        let mut gate = Gate::new(GatePolicy::Session);
        gate.wake(secs(1));
        assert_eq!(gate.end(secs(2)), None);
        assert_eq!(gate.tick(secs(10_000), false), None);
        assert_eq!(gate.route(), GateRoute::Vad);
    }
}
