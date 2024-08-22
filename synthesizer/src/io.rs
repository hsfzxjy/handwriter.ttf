use rten::InputOrOutput;
use rten_tensor::{self as rtt, AsView, Layout};
use std::sync::OnceLock;

trait ViewData<T> {
    fn view_data(&self) -> &[T];
}

impl<'a> ViewData<f32> for InputOrOutput<'a> {
    fn view_data(&self) -> &[f32] {
        use rten::Input::*;
        match self.as_input() {
            FloatTensor(t) => t.data().unwrap(),
            IntTensor(_) => unreachable!(),
        }
    }
}

#[inline]
fn zeros<const N: usize>(shape: [usize; N]) -> InputOrOutput<'static> {
    let t = rtt::Tensor::<f32>::zeros(&shape);
    rten::Output::FloatTensor(t).into()
}

pub struct HState {
    states: [InputOrOutput<'static>; HSTATE_SIZE],
}

const HSTATE_SIZE: usize = 8;

impl HState {
    fn new() -> Self {
        Self {
            states: [
                zeros([1, 1, 80]), // w
                zeros([1, 10]),    // k
                zeros([1, 400]),   // h1
                zeros([1, 400]),   // c1
                zeros([1, 400]),   // h2
                zeros([1, 400]),   // c2
                zeros([1, 400]),   // h3
                zeros([1, 400]),   // c3
            ],
        }
    }
    fn new_from(values: &mut impl Iterator<Item = rten::Output>) -> Self {
        Self {
            states: [
                values.next().unwrap().into(),
                values.next().unwrap().into(),
                values.next().unwrap().into(),
                values.next().unwrap().into(),
                values.next().unwrap().into(),
                values.next().unwrap().into(),
                values.next().unwrap().into(),
                values.next().unwrap().into(),
            ],
        }
    }
    fn as_feed(&self) -> impl Iterator<Item = InputOrOutput> {
        self.states.iter().map(|x| x.as_input().into())
    }
    pub fn swap(&mut self, other: &mut Self) {
        std::mem::swap(self, other)
    }
}

pub struct IdList {
    keys: &'static [&'static str],
    ids: OnceLock<Vec<usize>>,
}

impl IdList {
    const fn new(keys: &'static [&'static str]) -> Self {
        Self {
            keys,
            ids: OnceLock::new(),
        }
    }
    pub fn get(&self, model: &rten::Model) -> &[usize] {
        self.ids
            .get_or_init(|| {
                self.keys
                    .iter()
                    .map(|k| model.find_node(k).unwrap())
                    .collect()
            })
            .as_ref()
    }
}

static INPUT_IDS: IdList = IdList::new(&[
    "w.1", "k.1", "h1.1", "c1.1", "h2.1", "c2.1", "h3.1", "c3.1", //
    "x", "c", "bias",
]);

pub struct Input {
    pub x: rtt::NdTensor<f32, 3>,
    c: rtt::NdTensor<f32, 3>,
    bias: rtt::NdTensor<f32, 1>,
    pub hstate: HState,
}

impl Input {
    pub fn new(c: rtt::NdTensor<f32, 3>, bias: f32) -> Self {
        Self {
            x: rtt::NdTensor::zeros([1, 1, 3]),
            c,
            bias: rtt::NdTensor::full([1], bias),
            hstate: HState::new(),
        }
    }

    pub fn as_feed(&self, model: &rten::Model) -> Vec<(usize, InputOrOutput)> {
        INPUT_IDS
            .get(model)
            .iter()
            .cloned()
            .zip(self.hstate.as_feed().chain([
                self.x.view().into(),
                self.c.view().into(),
                self.bias.view().into(),
            ]))
            .collect()
    }
}

pub static OUTPUT_IDS: IdList = IdList::new(&[
    "w", "k", "h1", "c1", "h2", "c2", "h3", "c3", //
    "pi", "sd", "ro", "eos", "mu", "phi",
]);

pub struct Output {
    pi: InputOrOutput<'static>,
    sd: InputOrOutput<'static>,
    ro: InputOrOutput<'static>,
    eos: InputOrOutput<'static>,
    mu: InputOrOutput<'static>,
    phi: InputOrOutput<'static>,
    pub hstate: HState,
}

impl Output {
    pub fn new_from(outputs: Vec<rten::Output>) -> Self {
        let mut outputs = outputs.into_iter();
        let hstate = HState::new_from(&mut outputs);

        let pi = outputs.next().unwrap().into();
        let sd = outputs.next().unwrap().into();
        let ro = outputs.next().unwrap().into();
        let eos = outputs.next().unwrap().into();
        let mu = outputs.next().unwrap().into();
        let phi = outputs.next().unwrap().into();

        Self {
            hstate,
            pi,
            sd,
            ro,
            eos,
            mu,
            phi,
        }
    }
    pub fn sample_next_point(&self, rng: &mut rand::rngs::SmallRng) -> [f64; 3] {
        use rand::distributions::weighted::WeightedIndex;
        use rand::distributions::Distribution;
        let num_components = self.pi.size(0);
        let component = WeightedIndex::new(self.pi.view_data()).unwrap().sample(rng);
        let mu1 = self.mu.view_data()[component];
        let mu2 = self.mu.view_data()[component + num_components];
        let sd1 = self.sd.view_data()[component];
        let sd2 = self.sd.view_data()[component + num_components];
        let ro = self.ro.view_data()[component];
        let sample = crate::myrand::Bv2dNormal {
            mu: [mu1 as f64, mu2 as f64],
            sigma: [sd1 as f64, sd2 as f64],
            ro: ro as f64,
        }
        .sample(rng);
        [
            sample[0],
            sample[1],
            if self.eos.view_data()[0] > 0.5 {
                1.
            } else {
                0.
            },
        ]
    }
    pub fn is_end_of_string(&self) -> bool {
        let (last_phi, rest_phi) = self.phi.view_data().split_last().unwrap();
        let last_phi = *last_phi;
        if last_phi > 0.8 {
            return true;
        }
        if self.eos.view_data()[0] <= 0.5 {
            return false;
        }
        for x in rest_phi {
            if *x > last_phi {
                return false;
            }
        }
        return true;
    }
}
