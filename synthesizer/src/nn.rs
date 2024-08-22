use std::{
    cell::OnceCell,
    num::NonZeroUsize,
    sync::{Mutex, MutexGuard, OnceLock},
};

use rand::SeedableRng;
use rten::Model;
use rten_tensor::Layout;

use crate::{io, tokenizer};

struct ResultCache;
type LRUCache = lru::LruCache<Vec<u8>, Vec<[f64; 3]>>;

impl ResultCache {
    fn _access<'a>() -> MutexGuard<'a, OnceCell<LRUCache>> {
        static CACHE: Mutex<OnceCell<LRUCache>> = Mutex::new(OnceCell::new());
        let c = CACHE.lock().unwrap();
        c.get_or_init(|| LRUCache::new(NonZeroUsize::new(10).unwrap()));
        c
    }
    fn get(key: &[u8]) -> Option<Vec<[f64; 3]>> {
        Self::_access().get_mut().unwrap().get(key).cloned()
    }
    fn set(key: &[u8], value: Vec<[f64; 3]>) {
        Self::_access()
            .get_mut()
            .unwrap()
            .push(key.to_owned(), value);
    }
}

struct NormParam {
    sd: f64,
    mu: f64,
}
const NORMS: [NormParam; 2] = [
    NormParam {
        sd: 42.34926223754883,
        mu: 8.217162132263184,
    },
    NormParam {
        sd: 37.07347869873047,
        mu: 0.1212363988161087,
    },
];

fn load_model() -> &'static Model {
    static MODEL_BYTES: &[u8] = include_bytes!("../../weights/synthesis_network_52.rten");
    static MODEL: OnceLock<rten::Model> = OnceLock::new();

    MODEL.get_or_init(|| {
        rten::ModelOptions::with_all_ops()
            .enable_optimization(false)
            .load_static_slice(&MODEL_BYTES)
            .expect("fail to load model")
    })
}

pub struct Generator {
    model: &'static rten::Model,
    rng: rand::rngs::SmallRng,
}

impl Generator {
    pub fn new() -> Self {
        Self {
            model: load_model(),
            rng: rand::rngs::SmallRng::from_seed({
                let mut seed = <rand::rngs::SmallRng as SeedableRng>::Seed::default();
                seed.fill(42);
                seed
            }),
        }
    }
    pub fn generate_for(&mut self, text: &[u8]) -> Vec<[f64; 3]> {
        if let Some(res) = ResultCache::get(text) {
            return res;
        }

        let c = tokenizer::prepare_string(text);
        let u = c.shape()[1];
        let steps = text.len() * 35;
        let mut input = io::Input::new(c, 0.85);
        let mut prev_x = 0f64;
        let mut prev_y = 0f64;
        let mut points = vec![];
        let output_ids = io::OUTPUT_IDS.get(&self.model);
        for _ in 0..steps {
            let results = self
                .model
                .run(input.as_feed(self.model), &output_ids, None)
                .expect("fail to run");
            let mut output = io::Output::new_from(results);
            let mut x_temp = output.sample_next_point(&mut self.rng);
            let xdenorm = x_temp[0] * NORMS[0].sd + NORMS[0].mu;
            let ydenorm = x_temp[1] * NORMS[1].sd + NORMS[1].mu;
            prev_x += xdenorm;
            prev_y += ydenorm;
            if u >= 6 && output.is_end_of_string() {
                x_temp[2] = 1.0;
                points.push([prev_x, prev_y, x_temp[2]]);
                break;
            }
            points.push([prev_x, prev_y, x_temp[2]]);
            let x_temp = [x_temp[0] as f32, x_temp[1] as f32, x_temp[2] as f32];
            input.x.data_mut().unwrap().copy_from_slice(&x_temp[..]);
            input.hstate.swap(&mut output.hstate);
        }
        ResultCache::set(text, points.clone());
        points
    }
}
