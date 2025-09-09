#!/usr/bin/env python3
"""
Run one request with lmdeploy and make sure something is printed.
Supports TurboMind or PyTorch engine.
"""

import argparse
import asyncio
import time
from typing import List

from lmdeploy.messages import GenerationConfig, PytorchEngineConfig, TurbomindEngineConfig
from lmdeploy.tokenizer import Tokenizer
from lmdeploy.utils import get_logger

get_logger('lmdeploy').setLevel('DEBUG')

async def run_once(model, tokenizer, prompt_ids: List[int], gen_cfg: GenerationConfig):
    chatbot = model.create_instance()
    session_id = 0
    new_ids: List[int] = []

    # ==== perf: 初始化计时 ====
    t0 = time.perf_counter()     # 端到端起点
    ttft = None                  # time-to-first-token
    total_new = 0                # 累计新 token 数
    dec_time = 0.0               # 解码耗时

    async for out in chatbot.async_stream_infer(
            session_id,
            input_ids=prompt_ids,
            gen_config=gen_cfg,
            sequence_start=True,
            sequence_end=True,
            stream_output=True):
        # ① 只关心新出的 token 数
        new_n = out.num_token         # 本轮新增 token 数
        if new_n == 0:
            continue

        # ② 首 token 到达时间
        if ttft is None:
            ttft = time.perf_counter()

        total_new = new_n

        # ③ 取增量 id 并手动解码（保持你的原意：只保留最后一批的 ids）
        new_ids = out.token_ids

    t_stream_end = time.perf_counter()  # 流式结束时间

    # 兼容 PyTorch Engine 手动结束
    if hasattr(chatbot, "end"):
        await chatbot.async_end(session_id)

    print(new_ids)

    # ---- 非流式兜底打印 ----
    if new_ids:
        t_dec0 = time.perf_counter()
        # 去掉 prompt 部分，只解码新生成的 token（保持你的原逻辑）
        text = tokenizer.decode(new_ids, skip_special_tokens=False)
        t_dec1 = time.perf_counter()
        dec_time = t_dec1 - t_dec0
        print("\n\n==== decoded once more ====\n" + text)

    # ==== perf: 打印统计 ====
    e2e = t_stream_end - t0
    ttft_val = (ttft - t0) if ttft is not None else float('nan')
    gen_time = (t_stream_end - ttft) if ttft is not None else 0.0
    gen_tps = (total_new / gen_time) if gen_time > 0 else 0.0

    print(
        "[perf] prompt=%d toks, gen=%d toks | e2e=%.3fs, ttft=%.3fs, gen_time=%.3fs, gen_tps=%.2f tok/s, decode=%.3fs"
        % (len(prompt_ids), total_new, e2e, ttft_val, gen_time, gen_tps, dec_time)
    )

def build_engine(path: str, backend: str, tp: int, ep: int, dtype: str, tok):
    if backend == "turbomind":
        from lmdeploy.turbomind import TurboMind
        cfg = TurbomindEngineConfig(tp=tp, ep=ep, dtype=dtype)
        return TurboMind.from_pretrained(path, tokenizer=tok, engine_config=cfg)
    else:
        from lmdeploy.pytorch.engine import Engine
        cfg = PytorchEngineConfig(tp=tp, ep=ep, dtype=dtype)
        return Engine(path, tokenizer=tok, engine_config=cfg)


def parse_args():
    p = argparse.ArgumentParser("minimal lmdeploy run")
    p.add_argument("model_path")
    p.add_argument("--backend", choices=["turbomind", "pytorch"], default="turbomind")
    p.add_argument("--prompt", default="What is machine learning?")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--ep", type=int, default=1)
    p.add_argument("--dtype", default="auto")
    return p.parse_args()


def main():
    args = parse_args()
    tokenizer = Tokenizer(args.model_path)
    # prompt = 'You are a shopping guide AI assistant in Shopee e-commerce platform. Now given the user\'s original question and 3 related items info.\nPlease give a brief shopping suggestions to the user\'s question at first, then rename the items with corresponding reasons in provided order.\n\n[The start of item info]\nThe first item info dict is : {\'item_name\': \'Yeors -  Long Cullote Rosie Pants - Celana Murah Berkualitas Kulot Rib Knit Premium - Korean Loose Pants Trousers Wanita - Loungewear - Celana Kulot XL Perempuan Murah Berkualitas - Petite Cozie Pants\', \'item_description\': \'Limited quantity and SELLING FAST! Super bagus dan juga our best seller, kamu wajib punya!! Jangan mikir dua kali , takut keburu sold out 😜\\n• Khusus warna PUTIH , pasti akan NERAWANG ‼️ Jadi direkomendasikan untuk memakain daleman celana dalem seamless warna kulit 🙏🏻\\n• Tidak menerima penukaran warna maupun ukuran, mohon dipastikan ukuran dan warna yang di inginkan sudah sesuai saat Check Out 🙏🏻🙏🏻\\nINFO PENTING: Celana ini bahan  Rib Knit ya guys jadi mohon untuk tidak di cuci menggunakan mesin cuci terutama yang ada pengeringnya, bisa meyebabkan kainnya menciut atau rusak. Hanya boleh dicuci pelanpelan menggunakan tangan 🙏🏻 \\nTidak menerima complainan bila celana sudah di cuci dan pesnanan sudah diselesaikan\\nWarna:  Black, Broken White, Cream, Electric Blue, Fuschia, Olive, Soft Pink \\n• Real pict!! Karena perbedaan resolusi warna tiap layar berbeda, jadi warna tidak bisa 100% sesuai foto, tapi sudah kami usahakan mirip sesuai dengan aslinya 🙏🏻 \\nDetail Produk : \\n• Material : Rib Knit premium (Tebal, Lembut, Tidak nerawang)\\n• Ukuran : 2 ukuran ( Small and Reguler )\\n• Salur Kecil\\n• Menggunakan pengait dan restleting YKK\\n• Pinggang ada karet di belakang jadi bisa melar\\n• Ukuran celana "SMALL" untuk small and petite girl, jadi tidak direkomendasikan untuk yang punya pinggang, paha dan pinggul besar CO yang size "SMALL" , rekomendasi maksimal untuk bb 40-50 kg aja \\n • Ukuran celana "REGULER" direkomendasikan untuk yang punya pinggang, paha dan pinggul besar , rekomendasi maksimal untuk bb 55-65 kg \\nDetail Ukuran "SMALL" (S) :\\nPanjang 101 cm\\nLingkar pinggul 84-100 cm\\nLingkar Pinggang 60-80 cm\\nLingkar Paha 56-64 cm\\nPesak 72 cm\\nLingkar Kaki 46 cm\\nRekomendasi 40-52 kg\\nDetail Ukuran "REGULER" (M) :\\nPanjang 103 cm\\nLingkar pinggul 88-104 cm\\nLingkar Pinggang 65-96 cm\\nLingkar Paha 58-66 cm\\nPesak 74 cm\\nLingkar Kaki 50 cm\\nRekomendasi 53-63 kg\\nDetail Ukuran "LARGE" (L) :\\nPanjang 103 cm\\nLingkar pinggul 102-114 cm \\nLingkar Pinggang 70-102 cm\\nLingkar Paha 60-68 cm \\nPesak 78 cm\\nLingkar Kak\', \'price\': \'72000 IDR\'}\nThe second item info dict is : {\'item_name\': \'DIOBEE - DAILY JENNIE PANTS KNIT PREMIUM CULLOTE HIGHWAIST\', \'item_description\': \'DIOBEE - DAILY JENNIE PANTS KNIT PREMIUM CULLOTE HIGHWAIST\\n• JENNIE Knit Premium Knit Model terbaru dengan Serat lebih bagus & Berkualitas .\\nModel serat kayu yang buat lebih bagus dari Knit lainnya .\\nDETAIL PRODUK : \\n• LEMBUT, JATUH, FLOWY, TIDAK NERAWANG\\n• PINGGANG FULL KARET BISA STRECTH JADI JANGAN TAKUT KEKECILAN \\n• TERSEDIA 2 SAKU (KANAN-KIRI)\\n• TERSEDIA 2 KANCING DEPAN\\n• RESLETING DEPAN\\n• PREMIUM & HIGH QUALITY\\nDetail Size :\\nREMAJA S\\n( Muat BB 30 - 40Kg )\\n* Panjang celana -+ 95cm\\n* lingkar pinggang. +-50-87cm\\n* Lingkar paha +- 50-65cm\\n* Pisak +-40cm\\nDETAIL SIZE :\\nREMAJA M\\n( Muat BB 41-50kg  )\\n * Panjang celana -+ 100cm\\n* lingkar pinggang. +-55-90cm\\n* Lingkar paha +- 55-70cm\\n* Pisak +-40cm\\nDETAIL SIZE :\\nDEWASA XL\\n ( Muat BB 51-65kg )\\n* Panjang celana -+ 100cm\\n* lingkar pinggang. +-60-93cm\\n* Lingkar paha +- 60-75cm\\n* Pisak +-40cm\\nKOMPLAIN/RETUR\\n- Sertakan video unboxing untuk klaim komplain, apabila tidak menyertakan video saat komplain mohon maaf kami tidak dapat memproses komplain tersebut. Video unboxing wajib dari awal paket diterima atau sebelum dibuka sampai dengan selesai.\\n- Chat admin terlebih dahulu sebelum memberikan rating bintang apabila ada masalah pada paket yang diterima.\\n- Batas retur barang 3x24 jam setelah paket diterima.\\nPEMESANAN\\n- Mohon maaf untuk penukaran produk/size tidak diperbolehkan dengan alasan apapun, dikarenakan stok by sistem dan detail ukuran di deskripsi sudah cukup jelas.\\n- Produk yang dikirim sesuai dengan rincian pesanan dan tidak bisa menerima request warna/ukuran di note/catatan pesan.\\n- Cara pengukuran panjang celana yaitu dengan cara dipegang dengan kedua tangan bukan diletakkan dibawah.\\n⚠️ CHECKOUT = SETUJU DENGAN SEMUA KETENTUAN DIATAS ⚠️\', \'price\': \'48333 IDR\'}\nThe third item info dict is : {\'item_name\': \'ASOKA Loose Pants 999 Trousers Wanita [PART 1] - Celana Kulot Wanita - Celana Kantor Formal/Casual\', \'item_description\': \'- Celana dengan model pinggang super highwaist dan aman untuk butt besar\\n- Model pinggang belakang ada karet , bagian depan kaitan, dengan sleting jepang\\n- Celana yang multifungsi, bisa dipakai kerja, casual look, or korean look\\n- Untuk mendapatkan look rapih seperti foto dietalase, bisa disetrika terlebih dahulu dibagian tulang jahitan.\\nBahan : Katun Polyester Premium\\nDetail ukuran:\\n    • ALL SIZE (S-M)\\n- Lingkar pinggang : 58cm - 84cm\\n- Lingkar pinggul : 104cm\\n- Lingkar paha : 60cm\\n- Pisak : 34cm\\n- Lebar bawah : 28cm\\n- Panjang : 100cm\\n   • BIG SIZE (L-XL)\\n- Lingkar pinggang : 66cm - 94cm\\n- Lingkar pinggul : +-110cm\\n- Lingkar paha : 66cm\\n- Panjang : 100cm\\n- Pisak : 35cm\\n🔍 share and tag us @asokafashion.id\\nDISCLAIMER \\n- Jahitan tiap produk bisa saja tidak persis seperti detail diatas, ada toleransi ukuran 1-3cm. \\n- Mohon maaf, tidak menerima tukar warna maupun model.\\n- Membeli berarti setuju dengan semua ketentuan dari kami.\\nPENGIRIMAN\\n- Pengiriman dari Jakarta Barat, pemesanan sebelum jam 3 sore dikirim di hari yang sama.\\n- Pengiriman hanya Senin-Sabtu, tidak ada pengiriman pada hari Minggu/Hari Libur.\\nKOMPLAIN/RETUR\\n- Sertakan video unboxing untuk klaim komplain, apabila tidak menyertakan video saat komplain mohon maaf kami tidak dapat memproses komplain tersebut.\\n- Chat admin terlebih dahulu sebelum memberikan rating bintang apabila ada masalah pada paket yang diterima.\\n- Batas retur barang 3x24 jam setelah paket diterima.\\n*khusus warna BROKEN WHITE dan CREAM cenderung nerawang, sewajarnya jadi disarankan memakai dalaman senada atau tambahan celana legging didalem.\\n*khusus warna sage dan coksu, foto on model agak sedikit berbeda dari aslinya karna efek pantulan cahaya pada saat foto.\\n*bedanya 999 dan 998 hanya pada bagian kancing dan pengait saja.\', \'price\': \'89000 IDR\'}\n[The end of item info]\n\nuser\'s question: Tolong rekomendasikan celana wanita Korea dengan ulasan lebih banyak\nreverse_item_answer:'
    prompt_ids = tokenizer.encode(args.prompt)

    gen_cfg = GenerationConfig(max_new_tokens=args.max_new_tokens, ignore_eos=True)

    model = build_engine(args.model_path, args.backend, args.tp, args.ep, args.dtype, tokenizer)

    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(run_once(model, tokenizer, prompt_ids, gen_cfg))

    print("\n\n==== end main function ====\n")

    model.close()


if __name__ == "__main__":
    main()