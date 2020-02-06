use std::sync::Arc;
use std::cell::RefCell;
use primitives::{U256, H256};
use sr_primitives::generic::BlockId;
use sr_primitives::traits::{
	Block as BlockT, Header as HeaderT, ProvideRuntimeApi, UniqueSaturatedInto,
};
use hex;
use std::mem::transmute;
use client::{blockchain::HeaderBackend, backend::AuxStore};
use codec::{Encode, Decode};
use consensus_poc::PocAlgorithm;
use consensus_poc_primitives::{Seal as RawSeal, DifficultyApi,NonceData as RawNonceData};
use conjugatepoc_primitives::{Difficulty, AlgorithmApi, DAY_HEIGHT, HOUR_HEIGHT,HASH_SIZE,NONCE_SIZE,HASH_CAP,MESSAGE_SIZE};
use lru_cache::LruCache;
use rand::{SeedableRng, thread_rng, rngs::SmallRng};
use log::*;
mod shabal256;
use std::path::Path;
use std::fs;
use std::fs::OpenOptions;
use std::io::{Read, Seek, SeekFrom};
use crate::shabal256::{shabal256_deadline_fast, shabal256_hash_fast};
use jsonrpc_core::*;
use serde_derive::Deserialize;
use jsonrpc_http_server::*;
use jsonrpc_http_server::{cors::AccessControlAllowHeaders, hyper, RestApi, ServerBuilder};

const SCOOP_SIZE: usize = 64;

#[derive(Clone, PartialEq, Eq, Encode, Decode, Debug)]
pub struct Seal {
	pub difficulty: Difficulty,
	pub work: H256,
	pub nonce: H256,
}

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

#[derive(Clone, PartialEq, Eq, Encode, Decode, Debug)]
pub struct NonceData {
	pub height: u64,
	// pub base_target: u64,
	pub deadline: u64,
	pub nonce: u64,
	pub reader_task_processed: bool,
	pub account_id: u64,
	pub generation_sig: H256,
}

#[derive(Clone, PartialEq, Eq, Encode, Decode, Debug)]
pub struct Calculation {
	pub difficulty: Difficulty,
	pub pre_hash: H256,
	pub nonce: H256,
}

#[derive(Clone, PartialEq, Eq)]
pub struct Compute {
	pub key_hash: H256,
	pub pre_hash: H256,
	pub difficulty: Difficulty,
	pub nonce: H256,
}

#[derive(Deseriable)]
struct PocmineParams {
	pub generation_sig: H256,
	pub nonce: u64,
	pub base_target: U256,
	pub deadline: u64,
	pub height: u64,
}

thread_local!(static MACHINES: RefCell<LruCache<H256, randomx::FullVM>> = RefCell::new(LruCache::new(3)));

impl Compute {
	pub fn compute(self) -> Seal {
		MACHINES.with(|m| {
			let mut ms = m.borrow_mut();
			let calculation = Calculation {
				difficulty: self.difficulty,
				pre_hash: self.pre_hash,
				nonce: self.nonce,
			};

			let work = if let Some(vm) = ms.get_mut(&self.key_hash) {
				vm.calculate(&calculation.encode()[..])
			} else {
				let mut vm = randomx::FullVM::new(&self.key_hash[..]);
				let work = vm.calculate(&calculation.encode()[..]);
				ms.insert(self.key_hash, vm);
				work
			};

			Seal {
				nonce: self.nonce,
				difficulty: self.difficulty,
				work: H256::from(work),
			}
		})
	}
}

fn is_valid_hash(hash: &H256, difficulty: Difficulty) -> bool {
	let num_hash = U256::from(&hash[..]);
	let (_, overflowed) = num_hash.overflowing_mul(difficulty);

	!overflowed
}

fn key_hash<B, C>(
	client: &C,
	parent: &BlockId<B>
) -> Result<H256, String> where
	B: BlockT<Hash=H256>,
	C: HeaderBackend<B>,
{
	const PERIOD: u64 = 2 * DAY_HEIGHT;
	const OFFSET: u64 = 2 * HOUR_HEIGHT;

	let parent_header = client.header(parent.clone())
		.map_err(|e| format!("Client execution error: {:?}", e))?
		.ok_or("Parent header not found")?;
	let parent_number = UniqueSaturatedInto::<u64>::unique_saturated_into(*parent_header.number());

	let mut key_number = parent_number.saturating_sub(parent_number % PERIOD);
	if parent_number.saturating_sub(key_number) < OFFSET {
		key_number = key_number.saturating_sub(PERIOD);
	}

	let mut current = parent_header;
	while UniqueSaturatedInto::<u64>::unique_saturated_into(*current.number()) != key_number {
		current = client.header(BlockId::Hash(*current.parent_hash()))
			.map_err(|e| format!("Client execution error: {:?}", e))?
			.ok_or(format!("Block with hash {:?} not found", current.hash()))?;
	}

	Ok(current.hash())
}

pub struct RandomXAlgorithm<C> {
	client: Arc<C>,
}

impl<C> RandomXAlgorithm<C> {
	pub fn new(client: Arc<C>) -> Self {
		Self { client }
	}
}

impl<B: BlockT<Hash=H256>, C> PocAlgorithm<B> for RandomXAlgorithm<C> where
	C: HeaderBackend<B> + AuxStore + ProvideRuntimeApi,
	C::Api: DifficultyApi<B, Difficulty> + AlgorithmApi<B>,
{
	type Difficulty = Difficulty;

	fn difficulty(&self, parent: &BlockId<B>) -> Result<Difficulty, String> {
		let difficulty = self.client.runtime_api().difficulty(parent)
			.map_err(|e| format!("Fetching difficulty from runtime failed: {:?}", e));
		info!("Next block's difficulty: {:?}", difficulty);
		difficulty
	}

	fn getmineinfo(&self){
		let mut io = IoHandler::new();
		io.add_method("get_mine_info",|| {
			futures::finished(Value::String(format!("mine info")))
		});
		let server = ServerBuilder::new(io)
			.threads(1)
			.rest_api(RestApi::Unsecure)
			.cors(DomainsValidation::AllowOnly(vec![AccessControlAllowOrigin::Any]))
			.start_http(&"127.0.0.1:3030".parse().unwrap())
			.expect("Unable to start RPC server");
    	server.wait();
	}

	fn verify(
		&self,
		parent: &BlockId<B>,
		pre_hash: &H256,
		seal: &RawSeal,
		difficulty: Difficulty,
	) -> Result<bool, String> {
		assert_eq!(self.client.runtime_api().identifier(parent)
				   .map_err(|e| format!("Fetching identifier from runtime failed: {:?}", e))?,
				   conjugatepoc_primitives::ALGORITHM_IDENTIFIER);

		let key_hash = key_hash(self.client.as_ref(), parent)?;

		let seal = match Seal::decode(&mut &seal[..]) {
			Ok(seal) => seal,
			Err(_) => return Ok(false),
		};

		if !is_valid_hash(&seal.work, difficulty) {
			return Ok(false)
		}

		let compute = Compute {
			key_hash,
			difficulty,
			pre_hash: *pre_hash,
			nonce: seal.nonce,
		};

		if compute.compute() != seal {
			return Ok(false)
		}

		Ok(true)
	}

	fn poc_verify(
		&self,
		parent: &BlockId<B>,
		pre_hash: &H256,
		nonce_data: &RawNonceData,
		baseTarget: Difficulty,
	) -> Result<bool, String> {
		assert_eq!(self.client.runtime_api().identifier(parent)
			.map_err(|e| format!("Fetching identifier from runtime failed: {:?}", e))?,conjugatepoc_primitives::ALGORITHM_IDENTIFIER);
		let nonce_data = match NonceData::decode(&mut &nonce_data[..]){
			Ok(nonce_data) => nonce_data,
			Err(_) => return Ok(false),
		};
		let account_id = nonce_data.account_id;
		let height = nonce_data.height;
		let nonce = nonce_data.nonce;
		let generation_sig = nonce_data.generation_sig;
		let submit_deadline = nonce_data.deadline;
		let gensig = decode_gensig(&generation_sig);
		let scoop = calculate_scoop(height,&gensig);
		let mut cache = vec![0u8; 262144];
		noncegen_rust(&mut cache[..], account_id, nonce, 1);
		let address = 64 * scoop as usize;
		let mirrorscoop = 4095 - scoop as usize;
		let mirroraddress = 64 * mirrorscoop as usize;
		println!("Verify function Hash 2: (PoC2)       : {:?}",&hex::encode(&cache[mirroraddress + 32..mirroraddress + 64]));
		let mut mirrorscoopdata = vec![0u8; 64];
    	mirrorscoopdata[0..32].clone_from_slice(&cache[address..address + 32]);
    	mirrorscoopdata[32..64].clone_from_slice(&cache[mirroraddress + 32..mirroraddress + 64]);
		let (deadline, _) = find_best_deadline_rust(&mirrorscoopdata[..], 1, &gensig);
		let deadline_adj = deadline / baseTarget.as_u64();
		println!("Verify Function Deadline PoC2 (raw)  : {}", deadline);
		println!("Verify Function Deadline PoC2 (adj)  : {}", deadline_adj);
		if submit_deadline == deadline {
			return Ok(true);
		}
		Ok(false)
	}

	fn poc_mine(
		&self,
		parent: &BlockId<B>,
		generation_sig: H256,
		baseTarget: U256,
	) -> Result<Option<RawNonceData>,String> {
		// PoC 挖矿，从钱包提交过来的rpc请求，submit_nonce方法，包括参数 SubmitNonce 结构体中的参数。
			// pub accout_id: u64, 钱包账号id，也是plot_id
			// pub nonce: u64,  nonce_number,scoop_number
			// pub height: u64, 下一个区块高度
			// pub block: u64,	区块id
			// pub deadline_unadjusted: u64, base_target
			// pub deadline: u64, 	
			// pub gen_sig: [u8;32],

		// 验证提交的挖矿数据
			// poc_verify

		// 查看deadline时间是否流逝完，是否可以出块

		let height : u64 = 1;
		// todo: get the height from the parent
		let filename = "/Users/mac/projects/conjugate-poc/10790126960500947771_1_4096";
		let plotfile = Path::new(&filename);
		if !plotfile.is_file() {
			return Ok(None);
		}
		let name = plotfile.file_name().unwrap().to_str().unwrap();
		let parts: Vec<&str> = name.split("_").collect();
		let account_id = parts[0].parse::<u64>().unwrap();
		let start_nonce = parts[1].parse::<u64>().unwrap();
		let nonces = parts[2].parse::<u64>().unwrap();
		let size = fs::metadata(plotfile).unwrap().len();
		let exp_size = nonces * 4096 * 64;
		if size != exp_size as u64 {
			println!("expected plot size {} but got {}",exp_size,size);
			return Ok(None);
		}
		let nonce = 1u64;
		if !(nonce >= start_nonce && nonce < start_nonce + nonces){
			println!("File does't contain requested nonce");
			return Ok(None);
		}
		let gensig = decode_gensig(&generation_sig);
		let scoop = calculate_scoop(height,&gensig);
		let address = scoop as u64 * 64 * 4096 + (nonce - start_nonce) * 64;
		let mut file = OpenOptions::new().read(true).open(plotfile).unwrap();
		let mut scoopdata = vec![0u8; 64];
		file.seek(SeekFrom::Start(address)).unwrap();
		file.read_exact(&mut scoopdata[0..64]).unwrap();
		println!("Hash 1:              : {:?}",&hex::encode(&scoopdata[0..32]));
        println!("Hash 2:              : {:?}",&hex::encode(&scoopdata[32..64]));
		let (deadline, best_offset) = find_best_deadline_rust(&scoopdata[..], 1, &gensig);
		let deadline_adj = deadline / baseTarget.as_u64();
		println!("Deadline 2 (raw)     : {}", deadline);
		println!("Deadline 2 (adj)     : {}", deadline_adj);

		let noncedata = NonceData{
			height,
			deadline,
			nonce: best_offset,
			reader_task_processed: true,
			account_id,
			generation_sig,
		};
		return Ok(Some(noncedata.encode()))
		// if deadline_adj <= targetDeadline {
		// 	let noncedata = NonceData{
		// 		height,
		// 		baseTarget,
		// 		deadline,
		// 		nonce: best_offset,
		// 		true,
		// 		account_id,
		// 		generation_sig,
		// 	};
		// 	return Ok(Some(noncedata.encode()))
		// }
		// Ok(None)
	}

	fn mine(
		&self,
		parent: &BlockId<B>,
		pre_hash: &H256,
		difficulty: Difficulty,
		round: u32,
	) -> Result<Option<RawSeal>, String> {
		let mut rng = SmallRng::from_rng(&mut thread_rng())
			.map_err(|e| format!("Initialize RNG failed for mining: {:?}", e))?;
		let key_hash = key_hash(self.client.as_ref(), parent)?;

		for _ in 0..round {
			let nonce = H256::random_using(&mut rng);

			let compute = Compute {
				key_hash,
				difficulty,
				pre_hash: *pre_hash,
				nonce,
			};

			let seal = compute.compute();

			if is_valid_hash(&seal.work, difficulty) {
				return Ok(Some(seal.encode()))
			}
		}
		Ok(None)
	}
}

pub fn find_best_deadline_rust(data: &[u8],number_of_nonces: u64,gensig: &[u8;32]) -> (u64,u64){
	let mut best_deadline = std::u64::MAX;
	let mut best_offset = 0;
	for i in 0..number_of_nonces as usize {
		let result = shabal256_deadline_fast(&data[i * SCOOP_SIZE..i * SCOOP_SIZE + SCOOP_SIZE], &gensig);
		if result < best_deadline {
			best_deadline = result;
			best_offset = i;
		}
	}
	(best_deadline,best_offset as u64)
}
pub fn decode_gensig(gensig: &H256) -> [u8;32] {
	let mut gensig_bytes = [0;32];
	gensig_bytes[..].clone_from_slice(&hex::decode(gensig).unwrap());
	gensig_bytes
}
pub fn calculate_scoop(height: u64, gensig: &[u8;32]) -> u32 {
	let mut data: [u8;64] = [0;64];
	let height_bytes: [u8;8] = unsafe {transmute(height.to_be())};
	data[..32].clone_from_slice(gensig);
	data[32..40].clone_from_slice(&height_bytes);
	data[40] = 0x80;
	let data = unsafe { std::mem::transmute::<&[u8; 64], &[u32; 16]>(&data) };
	let new_gensig = &shabal256_hash_fast(&[], &data);
	(u32::from(new_gensig[30] & 0x0F) << 8) | u32::from(new_gensig[31])
}

pub fn noncegen_rust(cache: &mut [u8], numeric_id: u64, local_startnonce: u64, local_nonces: u64) {
	let numeric_id: [u32; 2] = unsafe { std::mem::transmute(numeric_id.to_be()) };
	let mut final_buffer = [0u8; HASH_SIZE];

	// prepare termination strings
	let mut t1 = [0u32; MESSAGE_SIZE];
	t1[0..2].clone_from_slice(&numeric_id);
	t1[4] = 0x80;

	let mut t2 = [0u32; MESSAGE_SIZE];
	t2[8..10].clone_from_slice(&numeric_id);
	t2[12] = 0x80;

	let mut t3 = [0u32; MESSAGE_SIZE];
	t3[0] = 0x80;

	for n in 0..local_nonces {
		// generate nonce numbers & change endianness
		let nonce: [u32; 2] = unsafe { std::mem::transmute((local_startnonce + n).to_be()) };
		// store nonce numbers in relevant termination strings
		t1[2..4].clone_from_slice(&nonce);
		t2[10..12].clone_from_slice(&nonce);

		// start shabal rounds

		// 3 cases: first 128 rounds uses case 1 or 2, after that case 3
		// case 1: first 128 rounds, hashes are even: use termination string 1
		// case 2: first 128 rounds, hashes are odd: use termination string 2
		// case 3: round > 128: use termination string 3
		// round 1
		let hash = shabal256_hash_fast(&[], &t1);

		cache[n as usize * NONCE_SIZE + NONCE_SIZE - HASH_SIZE
			..n as usize * NONCE_SIZE + NONCE_SIZE]
			.clone_from_slice(&hash);
		let hash = unsafe { std::mem::transmute::<[u8; 32], [u32; 8]>(hash) };

		// store first hash into smart termination string 2
		t2[0..8].clone_from_slice(&hash);
		// round 2 - 128
		for i in (NONCE_SIZE - HASH_CAP + HASH_SIZE..=NONCE_SIZE - HASH_SIZE)
			.rev()
			.step_by(HASH_SIZE)
		{
			// check if msg can be divided into 512bit packages without a
			// remainder
			if i % 64 == 0 {
				// last msg = seed + termination
				let hash = &shabal256_hash_fast(
					&cache[n as usize * NONCE_SIZE + i..n as usize * NONCE_SIZE + NONCE_SIZE],
					&t1,
				);
				cache[n as usize * NONCE_SIZE + i - HASH_SIZE..n as usize * NONCE_SIZE + i]
					.clone_from_slice(hash);
			} else {
				// last msg = 256 bit data + seed + termination
				let hash = &shabal256_hash_fast(
					&cache[n as usize * NONCE_SIZE + i..n as usize * NONCE_SIZE + NONCE_SIZE],
					&t2,
				);
				cache[n as usize * NONCE_SIZE + i - HASH_SIZE..n as usize * NONCE_SIZE + i]
					.clone_from_slice(hash);
			}
		}

		// round 128-8192
		for i in (HASH_SIZE..=NONCE_SIZE - HASH_CAP).rev().step_by(HASH_SIZE) {
			let hash = &shabal256_hash_fast(
				&cache[n as usize * NONCE_SIZE + i..n as usize * NONCE_SIZE + i + HASH_CAP],
				&t3,
			);
			cache[n as usize * NONCE_SIZE + i - HASH_SIZE..n as usize * NONCE_SIZE + i]
				.clone_from_slice(hash);
		}

		// generate final hash
		final_buffer.clone_from_slice(&shabal256_hash_fast(
			&cache[n as usize * NONCE_SIZE + 0..n as usize * NONCE_SIZE + NONCE_SIZE],
			&t1,
		));

		// XOR with final
		for i in 0..NONCE_SIZE {
			cache[n as usize * NONCE_SIZE + i] ^= final_buffer[i % HASH_SIZE];
		}
	}
}

#[cfg(test)]
mod tests {
	use super::*;
	use conjugatepoc_primitives::{H256, U256};

	#[test]
	fn randomx_len() {
		assert_eq!(randomx::HASH_SIZE, 32);
	}

	#[test]
	fn randomx_collision() {
		let mut compute = Compute {
			key_hash: H256::from([210, 164, 216, 149, 3, 68, 116, 1, 239, 110, 111, 48, 180, 102, 53, 180, 91, 84, 242, 90, 101, 12, 71, 70, 75, 83, 17, 249, 214, 253, 71, 89]),
			pre_hash: H256::default(),
			difficulty: U256::default(),
			nonce: H256::default(),
		};
		let hash1 = compute.clone().compute();
		U256::one().to_big_endian(&mut compute.nonce[..]);
		let hash2 = compute.compute();
		assert!(hash1 != hash2);
	}
}
