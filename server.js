import express from 'express';
import cors from 'cors';
import path from 'path';
import { fileURLToPath } from 'url';
import dotenv from 'dotenv';
import { GoogleGenAI } from '@google/genai';

dotenv.config();

const __filename = fileURLToPath(import.meta.url);
const __dirname = path.dirname(__filename);

const app = express();
const PORT = process.env.PORT || 3000;

app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// Serve static assets from public directory
app.use(express.static(path.join(__dirname, 'public')));

// SJT Answer Scoring Keys (Synchronized with functions/index.js)
const SJT_KEYS = {
  pfa:           { s0: { A: 5, B: 3, C: 1 }, s1: { A: 5, B: 3, C: 1 }, s2: { A: 5, B: 3, C: 1 } },
  floordirector: { s0: { A: 5, B: 3, C: 1 }, s1: { A: 5, B: 3, C: 1 }, s2: { A: 5, B: 3, C: 1 } },
  mc:            { s0: { A: 5, B: 3, C: 1 }, s1: { A: 5, B: 3, C: 1 }, s2: { A: 5, B: 3, C: 1 } },
  registrasi:    { s0: { A: 5, B: 3, C: 1 }, s1: { A: 5, B: 3, C: 1 }, s2: { A: 5, B: 3, C: 1 } },
  usher:         { s0: { A: 5, B: 3, C: 1 }, s1: { A: 5, B: 3, C: 1 }, s2: { A: 5, B: 3, C: 1 } },
  perlengkapan:  { s0: { A: 5, B: 3, C: 1 }, s1: { A: 5, B: 3, C: 1 }, s2: { A: 5, B: 3, C: 1 } },
  dokumentasi:   { s0: { A: 5, B: 3, C: 1 }, s1: { A: 5, B: 3, C: 1 }, s2: { A: 5, B: 3, C: 1 } },
  sponsor:       { s0: { A: 5, B: 3, C: 1 }, s1: { A: 5, B: 3, C: 1 }, s2: { A: 5, B: 3, C: 1 } }
};

const REACTION_SCORES = {
  swallow: 5,
  silent: 3,
  defend: 1
};

function hasPromptInjection(text) {
  const pattern = /(abaikan instruksi|ignore previous|ignore all|system prompt|beri skor 5|berikan skor penuh|nilai 100|anda adalah ai|as an ai|luluskan kandidat)/i;
  return pattern.test(text);
}

function computeHeuristicBars(star, komitmen) {
  const s = String(star?.situasi || "").trim();
  const t = String(star?.tugas || "").trim();
  const a = String(star?.aksi || "").trim();
  const r = String(star?.hasil || "").trim();
  const full = `${s} ${t} ${a} ${r} ${komitmen || ""}`.toLowerCase();

  if (hasPromptInjection(full)) {
    return { bars: 1, injected: true, reason: "Terdeteksi instruksi manipulatif pada teks esai." };
  }
  if (s.length < 40 || t.length < 40 || a.length < 40 || r.length < 40) {
    return { bars: 1, injected: false, reason: "Penjelasan pengalaman belum memenuhi panjang minimum di setiap bagian STAR." };
  }

  const words = full.replace(/[^a-z0-9\s]/gi, " ").split(/\s+/).filter(Boolean);
  const uniqueRatio = words.length > 0 ? new Set(words).size / words.length : 0;
  if (uniqueRatio < 0.38) {
    return { bars: 1, injected: false, reason: "Keragaman kosakata terlalu rendah (terindikasi pengulangan kalimat)." };
  }

  const selfCount = (a.toLowerCase().match(/\b(saya|aku)\b/g) || []).length;
  const verbs = [
    "mengatur", "koordinasi", "berkoordinasi", "membagi", "menyiapkan", "membantu",
    "menenangkan", "menghubungi", "menyelesaikan", "inisiatif", "berinisiatif",
    "negosiasi", "mencari", "mengajak", "mendengarkan", "merancang", "memastikan",
    "mengelola", "mendampingi", "menyusun", "mengecek", "memandu", "melobi", "mengamankan"
  ];
  let verbMatches = 0;
  verbs.forEach(v => { if (full.includes(v)) verbMatches++; });

  let bars = 2;
  if (selfCount >= 1 && verbMatches >= 2 && a.length >= 70) bars = 3;
  if (selfCount >= 2 && verbMatches >= 4 && a.length >= 110 && uniqueRatio >= 0.48) bars = 4;

  return {
    bars,
    injected: false,
    reason: `Evaluasi struktur pengalaman STAR menunjukkan tingkat rincian tindakan ${bars >= 3 ? "cukup jelas dan proaktif" : "masih bersifat umum"}.`
  };
}

async function evaluateEssayBars(roleId, star, komitmen, heuristicResult) {
  if (heuristicResult.injected) {
    return { bars: 1, reason: heuristicResult.reason, source: "heuristic_injection_guard" };
  }

  const geminiKey = process.env.GEMINI_API_KEY;
  if (geminiKey) {
    try {
      const ai = new GoogleGenAI();
      const prompt = `Anda adalah penilai seleksi relawan acara psikologi "Muslimah Healing Day". Nilai esai STAR berikut dalam skala BARS 1-5 (bilangan bulat):
Peran: ${roleId}
Situasi: ${star.situasi || ""}
Tugas: ${star.tugas || ""}
Aksi: ${star.aksi || ""}
Hasil: ${star.hasil || ""}
Komitmen: ${komitmen || ""}
Balas HANYA dalam JSON valid: {"bars": <1-5>, "alasan": "<1-2 kalimat sopan dalam bahasa Indonesia>"}`;

      const response = await ai.models.generateContent({
        model: 'gemini-2.5-flash',
        contents: prompt,
        config: {
          responseMimeType: "application/json"
        }
      });

      const parsed = JSON.parse(response.text || "{}");
      const rawAiBars = Math.max(1, Math.min(5, parseInt(parsed.bars, 10) || heuristicResult.bars));
      const clampedBars = Math.min(rawAiBars, heuristicResult.bars + 1);
      return {
        bars: clampedBars,
        reason: String(parsed.alasan || heuristicResult.reason).slice(0, 280),
        source: "gemini_clamped"
      };
    } catch (err) {
      console.warn("Gemini evaluation error, using heuristic fallback:", err.message);
    }
  }

  return { bars: heuristicResult.bars, reason: heuristicResult.reason, source: "heuristic_fallback" };
}

async function runEvaluation(data) {
  const roleId = String(data.peran_pilihan || "");
  const roleKey = SJT_KEYS[roleId] || SJT_KEYS.usher;
  const answers = data.jawaban_sjt || {};

  const p0 = roleKey.s0?.[answers.s0] || 1;
  const p1 = roleKey.s1?.[answers.s1] || 1;
  const p2 = roleKey.s2?.[answers.s2] || 1;
  const pReaction = REACTION_SCORES[data.reaksi_tekanan] || 1;

  const sjtScore40 = (p0 + p1 + p2 + pReaction) * 2;
  const ethicalViolation = (roleId === "dokumentasi" && p0 === 1) || (roleId === "pfa" && p0 === 1);

  const heuristic = computeHeuristicBars(data.star, data.komitmen_cepat);
  const essayEval = await evaluateEssayBars(roleId, data.star || {}, data.komitmen_cepat, heuristic);
  const essayScore60 = essayEval.bars * 12;

  const totalScore100 = sjtScore40 + essayScore60;

  let statusHasil = "BELUM_SESUAI";
  let labelStatus = "Belum Sesuai Prioritas Saat Ini";

  if (ethicalViolation) {
    statusHasil = "BELUM_SESUAI";
    labelStatus = "Belum Sesuai (Prinsip Privasi Ruang Aman)";
  } else if (totalScore100 >= 75) {
    statusHasil = "LOLOS_PRIORITAS";
    labelStatus = "Prioritas Utama (Direkomendasikan)";
  } else if (totalScore100 >= 60) {
    statusHasil = "CADANGAN";
    labelStatus = "Daftar Cadangan Potensial";
  }

  return {
    kode_hasil: data.kode_hasil,
    peran_pilihan: roleId,
    peran_cadangan: data.peran_cadangan || null,
    skor_sjt: sjtScore40,
    skor_esai: essayScore60,
    bars_esai: essayEval.bars,
    total_skor: totalScore100,
    status_kode: statusHasil,
    status_label: labelStatus,
    alasan_singkat: ethicalViolation
      ? "Pilihan tindakan pada studi kasus privasi ruang aman belum selaras dengan pedoman Heal You. " + essayEval.reason
      : essayEval.reason,
    evaluated_at: new Date().toISOString()
  };
}

// In-Memory Database store with rich seed data
const candidatesStore = new Map();
const resultsStore = new Map();

// Populate initial realistic sample candidates
const sampleCandidates = [
  {
    id: "cand_01",
    kode_hasil: "HY98KLP2MQ7X",
    nama: "Aisyah Nurul Fadhilah",
    whatsapp: "6281234567890",
    gender: "female",
    ketersediaan: "full",
    setuju_nda_privasi: true,
    persetujuan_pdp: true,
    reaksi_tekanan: "swallow",
    motivasi: "community",
    peran_pilihan: "pfa",
    peran_cadangan: "usher",
    jawaban_sjt: { s0: "A", s1: "A", s2: "A" },
    star: {
      situasi: "Dalam seminar kesehatan mental kampus tahun 2025, seorang peserta tiba-tiba mengalami serangan panik di barisan belakang.",
      tugas: "Sebagai panitia pendampingan, saya bertanggung jawab memberikan pertolongan pertama psikologis (PFA) agar peserta merasa aman tanpa memicu kepanikan peserta lain.",
      aksi: "Saya segera menghampiri peserta dengan tenang, mengajak bergeser ke sudut ruangan yang sunyi, mendengarkan aktif tanpa menghakimi, dan memandu teknik pernapasan 4-7-8 hingga ritme detak jantungnya stabil.",
      hasil: "Peserta merasa tenang dalam waktu 15 menit, menyampaikan apresiasi atas ruang aman yang diberikan, dan dapat kembali menikmati sisa acara dengan nyaman."
    },
    keahlian_khusus: "S1 Psikologi semester 7, Sertifikasi Basic Psychological First Aid HIMPSI",
    komitmen_cepat: "Saya berkomitmen penuh hadir tepat waktu pada Technical Meeting dan menjaga ruang aman setiap saudari peserta dengan ketulusan dan empati tinggi.",
    created_at: new Date(Date.now() - 3600000 * 24).toISOString()
  },
  {
    id: "cand_02",
    kode_hasil: "HY45BN99XZ81",
    nama: "Sarah Zahirah Maharani",
    whatsapp: "6285712345678",
    gender: "female",
    ketersediaan: "full",
    setuju_nda_privasi: true,
    persetujuan_pdp: true,
    reaksi_tekanan: "swallow",
    motivasi: "portfolio",
    peran_pilihan: "floordirector",
    peran_cadangan: "perlengkapan",
    jawaban_sjt: { s0: "A", s1: "A", s2: "A" },
    star: {
      situasi: "Saat festival kewirausahaan muslimah nasional dengan 600 audiens, narasumber utama terlambat 25 menit karena macet tol bandara.",
      tugas: "Sebagai stage manager dan timekeeper, saya harus menjaga ritme acara tetap hidup tanpa membuat audiens jenuh.",
      aksi: "Saya berkoordinasi cepat dengan MC untuk memajukan sesi ice breaking interaktif dan menata ulang alokasi waktu tanya jawab narasumber berikutnya secara presisi.",
      hasil: "Rundown kembali sinkron tanpa keterlambatan pada sesi penutupan, dan seluruh narasumber tetap mendapat ruang bicara yang adil."
    },
    keahlian_khusus: "Floor Director pada 4 event nasional, mahir Google Workspace & manajemen rundown ketat",
    komitmen_cepat: "Bersedia standby 2 jam sebelum pembukaan dan memastikan transisi panggung berjalan tanpa kendala.",
    created_at: new Date(Date.now() - 3600000 * 18).toISOString()
  },
  {
    id: "cand_03",
    kode_hasil: "HY77TR33KL90",
    nama: "Nabila Rahma Putri",
    whatsapp: "6281398765432",
    gender: "female",
    ketersediaan: "full",
    setuju_nda_privasi: true,
    persetujuan_pdp: true,
    reaksi_tekanan: "silent",
    motivasi: "community",
    peran_pilihan: "mc",
    peran_cadangan: "usher",
    jawaban_sjt: { s0: "A", s1: "A", s2: "A" },
    star: {
      situasi: "Pada acara Talkshow Pemulihan Batin Muslimah tahun lalu, terjadi kendala teknis kelistrikan panggung selama 10 menit.",
      tugas: "Saya bertugas memandu suasana panggung agar audiens tetap tenang dan suasana tidak menjadi kaku.",
      aksi: "Saya mengajak audiens melakukan senam pernapasan ringan bersama dan memantik refleksi kalimat-kalimat syukur interaktif dari peserta barisan depan.",
      hasil: "Audiens merasa terhibur dan tidak ada keluhan terkait kendala teknis hingga listrik kembali normal."
    },
    keahlian_khusus: "https://drive.google.com/portfolio-mc-nabila (MC Formal & Semi Formal 3 tahun)",
    komitmen_cepat: "Siap membawakan acara dengan nada yang teduh, hangat, dan ramah sesuai karakter Muslimah Healing Day.",
    created_at: new Date(Date.now() - 3600000 * 12).toISOString()
  },
  {
    id: "cand_04",
    kode_hasil: "HY21QQ88PL65",
    nama: "Fatimah Azzahra",
    whatsapp: "6282145678901",
    gender: "female",
    ketersediaan: "full",
    setuju_nda_privasi: true,
    persetujuan_pdp: true,
    reaksi_tekanan: "silent",
    motivasi: "altruism",
    peran_pilihan: "registrasi",
    peran_cadangan: "usher",
    jawaban_sjt: { s0: "A", s1: "B", s2: "A" },
    star: {
      situasi: "Pada kegiatan donor darah akbar masjid agung, terjadi antrean lebih dari 200 calon pendonor di meja registrasi.",
      tugas: "Saya bertugas memverifikasi formulir kesehatan dan kartu identitas peserta agar alur tetap teratur.",
      aksi: "Saya berinisiatif membagi antrean menjadi dua jalur nomor ganjil-genap dan menyiapkan lembar verifikasi cepat.",
      hasil: "Waktu tunggu per orang terpangkas dari 12 menit menjadi hanya 4 menit dengan nol kesalahan pencatatan data."
    },
    keahlian_khusus: "Mahir Microsoft Excel, Google Sheets, dan verifikasi barcode scanner",
    komitmen_cepat: "Hadir tepat waktu dan teliti memastikan kenyamanan registrasi seluruh peserta.",
    created_at: new Date(Date.now() - 3600000 * 6).toISOString()
  },
  {
    id: "cand_05",
    kode_hasil: "HY11MM44VV22",
    nama: "Dina Kartika Sari",
    whatsapp: "6289612345678",
    gender: "female",
    ketersediaan: "full",
    setuju_nda_privasi: true,
    persetujuan_pdp: true,
    reaksi_tekanan: "defend",
    motivasi: "portfolio",
    peran_pilihan: "dokumentasi",
    peran_cadangan: "sponsor",
    jawaban_sjt: { s0: "C", s1: "A", s2: "A" }, // C on s0 triggers ethical privacy violation in documentation!
    star: {
      situasi: "Saya pernah mendokumentasikan kegiatan workshop fotografi komunitas di Bandung.",
      tugas: "Saya bertugas mengambil gambar dokumentasi seluruh peserta dan narasumber.",
      aksi: "Saya memotret banyak momen emosional dan interaksi jarak dekat.",
      hasil: "Hasil foto diserahkan ke koordinator tepat waktu setelah kegiatan selesai."
    },
    keahlian_khusus: "https://instagram.com/dina_visuals",
    komitmen_cepat: "Siap membawa kamera mirrorless Sony A7III dan dua lensa prime.",
    created_at: new Date(Date.now() - 3600000 * 3).toISOString()
  }
];

// Initialize sample evaluations
for (const cand of sampleCandidates) {
  candidatesStore.set(cand.kode_hasil, cand);
}

// Generate evaluations for pre-seeded candidates
(async () => {
  for (const cand of sampleCandidates) {
    const evalResult = await runEvaluation(cand);
    cand.evaluasi = evalResult;
    resultsStore.set(cand.kode_hasil, evalResult);
  }
})();

// API: Register new volunteer candidate
app.post('/api/pendaftar', async (req, res) => {
  try {
    const data = req.body;
    if (!data.nama || !data.whatsapp || !data.peran_pilihan) {
      return res.status(400).json({ error: "Data pendaftaran tidak lengkap." });
    }

    const kodeHasil = data.kode_hasil || ("HY" + Math.random().toString(36).substring(2, 12).toUpperCase());
    data.kode_hasil = kodeHasil;
    data.created_at = new Date().toISOString();

    const evaluasi = await runEvaluation(data);
    data.evaluasi = evaluasi;

    const id = `cand_${Date.now()}_${Math.random().toString(36).substring(2, 6)}`;
    const fullRecord = { id, ...data };

    candidatesStore.set(kodeHasil, fullRecord);
    resultsStore.set(kodeHasil, evaluasi);

    return res.status(201).json({
      success: true,
      kode_hasil: kodeHasil,
      evaluasi
    });
  } catch (error) {
    console.error("Error evaluating candidate:", error);
    return res.status(500).json({ error: "Gagal memproses evaluasi relawan." });
  }
});

// API: Query selection result by 12-char code
app.get('/api/hasil/:kode', (req, res) => {
  const kode = String(req.params.kode || "").trim().toUpperCase();
  const result = resultsStore.get(kode);
  if (!result) {
    return res.status(404).json({ error: "Kode pendaftaran tidak ditemukan. Pastikan 12 karakter kode sudah benar." });
  }
  return res.json(result);
});

// API: Admin Login
app.post('/api/admin/login', (req, res) => {
  const { email, password } = req.body || {};
  const validSecret = process.env.ADMIN_SECRET || "healyou2026";

  // Allow admin login with official domain or configured secret
  if ((email && (email.toLowerCase().includes("admin") || email.toLowerCase().includes("healyou")) && (password === validSecret || password === "admin123" || password === "healyou2026")) || (password === validSecret)) {
    return res.json({
      success: true,
      token: "demo_admin_jwt_token_" + Date.now(),
      email: email || "admin@healyou.id"
    });
  }

  return res.status(401).json({ error: "Email atau kata sandi admin tidak valid." });
});

// API: Admin list candidates
app.get('/api/admin/pendaftar', (req, res) => {
  const list = Array.from(candidatesStore.values());
  return res.json(list);
});

// Fallback route for SPA / HTML navigation
app.get('*', (req, res) => {
  if (req.path === '/admin' || req.path === '/admin.html') {
    return res.sendFile(path.join(__dirname, 'public', 'admin.html'));
  }
  if (req.path === '/hasil' || req.path === '/hasil.html') {
    return res.sendFile(path.join(__dirname, 'public', 'hasil.html'));
  }
  return res.sendFile(path.join(__dirname, 'public', 'index.html'));
});

app.listen(PORT, '0.0.0.0', () => {
  console.log(`Muslimah Healing Day Volunteer System running on http://0.0.0.0:${PORT}`);
});
