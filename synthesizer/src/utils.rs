use std::sync;

use harfbuzz_wasm::debug;

pub fn set_panic_hook() {
    static INIT: sync::Once = sync::Once::new();
    INIT.call_once(|| std::panic::set_hook(Box::new(hook_impl)))
}

fn hook_impl(info: &core::panic::PanicInfo) {
    let msg = info.to_string();
    let backtrace = std::backtrace::Backtrace::capture();
    debug(&format!("{:#?}\n{:?}", backtrace, msg));
}
