# VPHT Algorithm Visualization

A tool for visualizing and brute-forcing VPHT algorithm results.

This project is built using [egui](https://github.com/emilk/egui) and [eframe](https://github.com/emilk/egui/tree/master/crates/eframe).
It is based on the [eframe template](https://github.com/emilk/eframe_template/).

Live demo: https://kalokak.github.io/vpht-algo/

## Getting Started

### Prerequisites

- Rust toolchain (`rustup`)
- [Trunk](https://trunkrs.dev/) (for web build): `cargo install --locked trunk`
- Web target: `rustup target add wasm32-unknown-unknown`

### Running Locally

**Native:**
```sh
cargo run --release
```

**Web:**
```sh
trunk serve
```
Open `http://127.0.0.1:8080/index.html#dev` in a browser.

> Note: The `#dev` suffix bypasses service worker caching, ensuring the latest build is loaded.

## Acknowledgments

- [eframe_template](https://github.com/emilk/eframe_template/) for the initial template.
- [egui](https://github.com/emilk/egui) for the GUI framework.
