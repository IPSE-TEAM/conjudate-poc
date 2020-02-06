use pocjsonrpc::run_server;

fn main() {
    hyper::rt::run(run_server(&'127.0.0.1:8080'.parse().unwrap()));
}