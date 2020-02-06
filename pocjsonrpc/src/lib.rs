use serde::{Serialize, Deserialize, de::DeserializeOwned};
use hyper::{Body, Request, Response, header, service::service_fn, Server, Method, Client,StatusCode};
use futures::{future, Future, stream::Stream};
use chrono::{DateTime, Utc, Duration};
use rand::{thread_rng, Rng};
use hyper::service::service_fn_ok;
use hyper::client::HttpConnector;

type GenericError = Box<dyn std::error::Error + Send + Sync>;
type ResponseFuture = Box<dyn Future<Item=Response<Body>, Error=GenericError> + Send>;


#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitNonce {
    pub accout_id: u64,
    pub nonce: u64,
    pub height: u64,
    pub block: u64,
    pub deadline_unadjusted: u64,
    pub deadline: u64, 
    pub gen_sig: [u8;32],
}

#[derive(Serialize, Deserialize, Debug)]
pub struct SubmitNonceResponse {
    accout_id: u64,
    nonce: u64,
    height: u64,
    block: u64,
    deadline: u64,
    base_target: u64,
    result: String,
}

#[derive(Serialize, Deserialize, Debug)]
pub struct MiningInfoResponse {
    height: u64,
    generation_signature: [u8;32],
    base_target: u64,
}

fn api_response(req: Request<Body>,client: &Client<HttpConnector>) -> ResponseFuture {
    match (req.method(),req.uri().path()) {
        (&Method::GET,"/get_mining_info") => {

            Box::new(future::ok(Response::new(Body::from("mining info"))))
        },
        (&Method::POST,"/submit_nonce") => {
            let ok_responce = req.into_body().concat2().from_err().and_then(|entire_body|{
                let str = String::from_utf8(entire_body.to_vec())?;                
                let mut data : serde_json::Value = serde_json::from_str(&str)?;
                data["test"] = serde_json::Value::from("test_value");
                let json = serde_json::to_string(&data)?;
                println!("{}",json);
                let response = Response::builder()
                    .status(StatusCode::OK)
                    .header(header::CONTENT_TYPE, "application/json")
                    .body(Body::from(json))?;
                Ok(response)
            });
            Box::new(ok_responce)
            // map_request_to_response(req, |req: SubmitNonce| {
                // let submitnonceresponse = SubmitNonceResponse {
                //     req.accout_id,
                //     req.nonce,
                //     req.height,
                //     req.block,
                //     req.deadline,
                //     req.deadline_unadjusted,
                //     "ok",
                // }
                
            // })
            // Box::new(future::ok(Response::new(Body::from("submit nonce response"))))
        },
        _ => Box::new(future::ok(Response::new(Body::empty()))),
    }
}

fn map_request_to_response<Req, Res, T>(req: Request<Body>, transformation: T) -> ResponseFuture where
    Req: DeserializeOwned,
    Res: Serialize,
    T: Fn(Req) -> Res + Send + Sync + 'static
{
    Box::new(req.into_body().concat2().from_err().and_then(move |entire_body| {
        let req = serde_json::from_slice(entire_body.as_ref())?;
        let res = transformation(req);
        let string = serde_json::to_string(&res)?;
        Response::builder()
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(string))
            .map_err(|e| e.into())
    }))
}

// pub fn run_server(address: &std::net::SocketAddr) -> impl Future<Item=(), Error=()> {
//     Server::bind(address)
//         .serve(|| service_fn(api_response))
//         .map_err(|e| eprintln!("server error: {}", e))
// }
fn run_server() {
    // socket address
    let addr = ([127, 0, 0, 1], 3000).into();
    
    // A Service for the api_response function
    let new_svc = move || {
        let client = Client::new();
        service_fn(move |req| {
            api_response(req,&client)
        })
    };
    let server = Server::bind(&addr).serve(new_svc).map_err(|e| eprintln!("server error: {}",e));
    // run this server forever
    hyper::rt::run(server);
}
