WASM=synthesizer/pkg/synthesizer_bg.wasm
AOT=synthesizer/pkg/synthesizer_bg.aot
TTF_Final=bin/HW_Syn.ttf
TTF_Raw=Harfbuzz-WASM-Fantasy/bin/HB_WASM_Fantasy-Thin.ttf

.PHONY: run
run:
	IMAGE_NAME=hsfzxjy/harfbuzz-wasm-handwriting-synthesis bin/run-docker

.PHONY: build
build: $(TTF_Final)
	cd bin && docker build -t hsfzxjy/harfbuzz-wasm-handwriting-synthesis .
	docker push hsfzxjy/harfbuzz-wasm-handwriting-synthesis

.PHONY: debug
debug: $(TTF_Final)
	cd bin && docker build -t test__hw_syn .
	IMAGE_NAME=test__hw_syn bin/run-docker

.PHONY: font
font: $(TTF_Final)

$(TTF_Final): $(AOT)
	$(MAKE) -C Harfbuzz-WASM-Fantasy
	Harfbuzz-WASM-Fantasy/bin/otfsurgeon -i $(TTF_Raw) add -o $(TTF_Final) Wasm < $(AOT)

$(WASM): $(wildcard synthesizer/src/**/*) $(wildcard synthesizer/src/*) $(wildcard weights/*)
	cd synthesizer; wasm-pack build --target web

$(AOT): $(WASM)
	docker run --rm -it -u `id -u`:`id -g` \
		-v ./synthesizer/pkg:/pkg \
		hsfzxjy/harfbuzz-wasm-fantasy \
		/fantasy/wamrc \
		--opt-level=3 \
		--enable-segue=i32.load,f32.load,i32.store,f32.store \
		--enable-tail-call \
		-v=5 \
		-o /pkg/synthesizer_bg.aot \
		/pkg/synthesizer_bg.wasm

.PHONY: wasm
wasm:
	cd synthesizer; wasm-pack build --target web
	Harfbuzz-WASM-Fantasy/wasm-micro-runtime/wamr-compiler/build/wamrc --opt-level=3 --enable-segue=i32.load,f32.load,i32.store,f32.store --enable-tail-call -v=5 -o x.aot $(WASM)