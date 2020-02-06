# ipse-core

Proof-of-capacity blockchain built on
[Substrate](https://github.com/paritytech/substrate).

## Overview

Ipse-core is the underlying consensus layer of IPSE project, which is the basic version of the whole application chain. The function modules to be added in the future are all extended based on this core version.

Ipse-core is developed based on Substrate and will try some new consensus algorithms at the consensus layer, and is a consensus algorithm that can be combined with the storage disk. For example, the PoC consensus algorithm has been proved successful so far.

## Network Launch

The first launch attempt is on! We currently do not provide any official binary
release, so please compile the node by yourself, using the instructions below.

Launch attempt means that it's an experimental launch. We relaunch the network
when bugs are found. Otherwise, the current network becomes the mainnet.

Substrate contains a variety of features including smart contracts and
democracy. However, for initial launch of ipse-core, we plan to only enable basic
balance and transfer module. This is to keep the network focused, and reduce
risks in terms of stability and safety. Also note that initially the democracy
module is also disabled, meaning we'll be updating runtime via hard fork until
that part is enabled.

## Prerequisites

Clone this repo and update the submodules:

```bash
git clone https://github.com/IPSE-TEAM/ipse-core
cd ipse-core
git submodule update --init --recursive
```

Install Rust:

```bash
curl https://sh.rustup.rs -sSf | sh
```

Install required tools:

```bash
./scripts/init.sh
```

## Run

### Full Node

```bash
cargo run --release
```

### Mining

Install `subkey`:

```bash
cargo install --force --git https://github.com/paritytech/substrate subkey
```

Generate an account to use as the target for mining:

```bash
subkey --sr25519 --network=16 generate
```

Remember the public key, and pass it to node for mining. For example:

```
cargo run --release -- --validator --author 0x7e946b7dd192307b4538d664ead95474062ac3738e04b5f3084998b76bc5122d
```


This project is a side project by Wei Tang, and is not endorsed by Parity
Technologies.
