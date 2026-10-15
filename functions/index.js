const { onDocumentCreated } = require("firebase-functions/v2/firestore");
const admin = require("firebase-admin");
admin.initializeApp();
const db = admin.firestore();

// Kunci SJT Rahasia di Server (Tidak ada di HTML klien)
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
  const verbs = ["mengatur", "koordinasi", "berkoordinasi", "membagi", "menyiapkan", "membantu", "menenangkan", "menghubungi", "menyelesaikan", "inisiatif", "berinisiatif", "negosiasi", "mencari", "mengajak", "mendengarkan", "merancang", "memastikan", "mengelola", "mendampingi", "menyusun", "mengecek", "memandu", "melobi", "mengamankan"];
  let verbMatches = 0;
  verbs.forEach(v => { if (full.includes(v)) verbMatches++; });

  let bars = 2;
  if (selfCount >= 1 && verbMatches >= 2 && a.length >= 70) bars = 3;
  if (selfCount >= 2 && verbMatches >= 4 && a.length >= 110 && uniqueRatio >= 0.48) bars = 4;

  return { bars, injected: false, reason: `Evaluasi struktur pengalaman STAR menunjukkan tingkat rincian tindakan ${bars >= 3 ? "cukup jelas dan proaktif" : "masih bersifat umum"}.` };
}

async function evaluateEssayBars(roleId, star, komitmen, heuristicResult) {
  if (heuristicResult.injected) return { bars: 1, reason: heuristicResult.reason, source: "heuristic_injection_guard" };
  const apiKey = process.env.ANTHROPIC_KEY;
  if (!apiKey) return { bars: heuristicResult.bars, reason: heuristicResult.reason, source: "heuristic_only" };

  const prompt = `Anda adalah penilai seleksi relawan acara psikologi "Muslimah Healing Day". Nilai esai STAR berikut dalam skala BARS 1-5 (bilangan bulat):
Peran: ${roleId}
Situasi: ${star.situasi}
Tugas: ${star.tugas}
Aksi: ${star.aksi}
Hasil: ${star.hasil}
Komitmen: ${komitmen}
Balas HANYA dalam JSON valid: {"bars": <1-5>, "alasan": "<1-2 kalimat sopan dalam bahasa Indonesia>"}`;

  try {
    const resp = await fetch("https://api.anthropic.com/v1/messages", {
      method: "POST",
      headers: { "Content-Type": "application/json", "x-api-key": apiKey, "anthropic-version": "2023-06-01" },
      body: JSON.stringify({ model: "claude-3-5-haiku-latest", max_tokens: 220, temperature: 0.1, messages: [{ role: "user", content: prompt }] })
    });
    if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
    const json = await resp.json();
    const parsed = JSON.parse(json?.content?.[0]?.text || "{}");
    const rawAiBars = Math.max(1, Math.min(5, parseInt(parsed.bars, 10) || heuristicResult.bars));
    const clampedBars = Math.min(rawAiBars, heuristicResult.bars + 1);
    return { bars: clampedBars, reason: String(parsed.alasan || heuristicResult.reason).slice(0, 280), source: "anthropic_clamped" };
  } catch (err) {
    return { bars: heuristicResult.bars, reason: heuristicResult.reason, source: "heuristic_fallback" };
  }
}

exports.evaluateApplication = onDocumentCreated("pendaftar_relawan/{docId}", async (event) => {
  const data = event.data.data();
  if (!data) return;

  const roleId = String(data.peran_pilihan || "");
  const roleKey = SJT_KEYS[roleId] || SJT_KEYS.usher;
  const answers = data.jawaban_sjt || {};

  const p0 = roleKey.s0[answers.s0] || 1;
  const p1 = roleKey.s1[answers.s1] || 1;
  const p2 = roleKey.s2[answers.s2] || 1;
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

  const hasilObj = {
    kode_hasil: data.kode_hasil,
    peran_pilihan: roleId,
    peran_cadangan: data.peran_cadangan || null,
    skor_sjt: sjtScore40,
    skor_esai: essayScore60,
    bars_esai: essayEval.bars,
    total_skor: totalScore100,
    status_kode: statusHasil,
    status_label: labelStatus,
    alasan_singkat: ethicalViolation ? "Pilihan tindakan pada studi kasus privasi ruang aman belum selaras dengan pedoman Heal You. " + essayEval.reason : essayEval.reason,
    evaluated_at: admin.firestore.FieldValue.serverTimestamp()
  };

  await event.data.ref.update({ evaluasi: hasilObj });
  if (data.kode_hasil) {
    await db.collection("hasil_seleksi").doc(String(data.kode_hasil)).set(hasilObj);
  }
});
