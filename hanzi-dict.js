// Dictionnaire maison : découpe une suite de caractères en mots (le plus long mot connu
// d'abord) et donne le pinyin et le sens de chaque mot.
// Données : CFDICT (français, CC BY-SA 3.0) et CC-CEDICT (anglais, CC BY-SA 4.0), converties par ml/build_dict.py.

// Lecture la plus courante pour quelques caractères à plusieurs prononciations.
const USUAL = {
  了: "le", 的: "de", 着: "zhe", 么: "me", 吗: "ma", 呢: "ne", 吧: "ba", 啊: "a", 个: "gè", 们: "men",
  都: "dōu", 还: "hái", 和: "hé", 没: "méi", 得: "de", 为: "wèi", 只: "zhǐ", 长: "cháng", 行: "xíng",
  重: "zhòng", 便: "biàn", 会: "huì", 要: "yào", 少: "shǎo", 发: "fā", 将: "jiāng", 看: "kàn",
  种: "zhǒng", 当: "dāng", 空: "kōng", 数: "shù", 好: "hǎo", 大: "dà", 不: "bù", 一: "yī",
};

export async function loadDict(url = "/dict/zh.json") {
  const r = await fetch(url);
  if (!r.ok) throw new Error("dictionnaire introuvable");
  return r.json();
}

function reading(dict, word) {
  const rs = dict[word];
  if (!rs) return null;
  const pref = USUAL[word];
  return (pref && rs.find((r) => r[0] === pref)) || rs[0];
}

const first = (s) => (s || "").split("; ")[0];

function entry(dict, word) {
  const r = reading(dict, word);
  return {
    word,
    pinyin: r ? r[0] : "",
    meaning: r ? r[1] || r[2] : "",
    lang: r && !r[1] && r[2] ? "en" : "fr",
    readings: dict[word] || [],
  };
}
const hasFr = (dict, w) => (dict[w] || []).some((r) => r[1]);

// Renvoie { words: [{ word, pinyin, meaning, lang, readings }], extras, pinyin, gloss }.
// On prend le plus long mot connu ; s'il n'existe qu'en anglais (ex. 禁止吸烟 « No smoking! »),
// on préfère des mots plus courts traduits en français et l'expression anglaise va dans `extras`.
export function analyze(dict, text) {
  const chars = [...text];
  const words = [], extras = [];
  for (let i = 0; i < chars.length;) {
    const at = (n) => chars.slice(i, i + n).join("");
    let len = Math.min(8, chars.length - i);
    while (len > 1 && !dict[at(len)]) len--;
    if (len > 1 && !hasFr(dict, at(len))) {
      let fr = len - 1;
      while (fr > 1 && !hasFr(dict, at(fr))) fr--;
      if (hasFr(dict, at(fr))) {
        if (len >= 3) extras.push(entry(dict, at(len))); // expression figée utile à montrer
        len = fr;
      }
    }
    words.push(entry(dict, at(len)));
    i += len;
  }
  return {
    words,
    extras,
    pinyin: words.map((w) => w.pinyin.replace(/ /g, "")).filter(Boolean).join(" "),
    gloss: words.map((w) => first(w.meaning) || "?").join(" · "),
  };
}
