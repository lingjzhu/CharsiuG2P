# -*- coding: utf-8 -*-
"""Language names and codes -> the ISO 639-3 based tags the dictionaries use.

Dictionary tags are ISO 639-3 codes (WikiPron's choice), sometimes with a
variety: "eng-us", "por-br", "cmn-tw". Users may instead give an ISO 639-1
code ("de"), an ISO 639-2/B code ("ger"), or an English name ("German").
Macrolanguage codes fall back to their members ("zho" -> "cmn", "yue", ...).
"""

from typing import Dict, List

# ISO 639-1 code, ISO 639-3 code, English name.
_ISO1 = """
aa aar Afar|ab abk Abkhazian|af afr Afrikaans|ak aka Akan|am amh Amharic|an arg Aragonese|
ar ara Arabic|as asm Assamese|av ava Avaric|ay aym Aymara|az aze Azerbaijani|ba bak Bashkir|
be bel Belarusian|bg bul Bulgarian|bi bis Bislama|bm bam Bambara|bn ben Bengali|bo bod Tibetan|
br bre Breton|bs bos Bosnian|ca cat Catalan|ce che Chechen|ch cha Chamorro|co cos Corsican|
cr cre Cree|cs ces Czech|cu chu Church Slavonic|cv chv Chuvash|cy cym Welsh|da dan Danish|
de deu German|dv div Dhivehi|dz dzo Dzongkha|ee ewe Ewe|el ell Greek|en eng English|
eo epo Esperanto|es spa Spanish|et est Estonian|eu eus Basque|fa fas Persian|ff ful Fula|
fi fin Finnish|fj fij Fijian|fo fao Faroese|fr fra French|fy fry Western Frisian|ga gle Irish|
gd gla Scottish Gaelic|gl glg Galician|gn grn Guarani|gu guj Gujarati|gv glv Manx|ha hau Hausa|
he heb Hebrew|hi hin Hindi|ho hmo Hiri Motu|hr hrv Croatian|ht hat Haitian Creole|hu hun Hungarian|
hy hye Armenian|hz her Herero|ia ina Interlingua|id ind Indonesian|ie ile Interlingue|ig ibo Igbo|
ii iii Nuosu|ik ipk Inupiaq|io ido Ido|is isl Icelandic|it ita Italian|iu iku Inuktitut|
ja jpn Japanese|jv jav Javanese|ka kat Georgian|kg kon Kongo|ki kik Kikuyu|kj kua Kwanyama|
kk kaz Kazakh|kl kal Greenlandic|km khm Khmer|kn kan Kannada|ko kor Korean|kr kau Kanuri|
ks kas Kashmiri|ku kur Kurdish|kv kom Komi|kw cor Cornish|ky kir Kyrgyz|la lat Latin|
lb ltz Luxembourgish|lg lug Ganda|li lim Limburgish|ln lin Lingala|lo lao Lao|lt lit Lithuanian|
lu lub Luba-Katanga|lv lav Latvian|mg mlg Malagasy|mh mah Marshallese|mi mri Maori|mk mkd Macedonian|
ml mal Malayalam|mn mon Mongolian|mr mar Marathi|ms msa Malay|mt mlt Maltese|my mya Burmese|
na nau Nauru|nb nob Norwegian Bokmal|nd nde North Ndebele|ne nep Nepali|ng ndo Ndonga|nl nld Dutch|
nn nno Norwegian Nynorsk|no nor Norwegian|nr nbl South Ndebele|nv nav Navajo|ny nya Chichewa|
oc oci Occitan|oj oji Ojibwe|om orm Oromo|or ori Odia|os oss Ossetian|pa pan Punjabi|pi pli Pali|
pl pol Polish|ps pus Pashto|pt por Portuguese|qu que Quechua|rm roh Romansh|rn run Kirundi|
ro ron Romanian|ru rus Russian|rw kin Kinyarwanda|sa san Sanskrit|sc srd Sardinian|sd snd Sindhi|
se sme Northern Sami|sg sag Sango|si sin Sinhala|sk slk Slovak|sl slv Slovenian|sm smo Samoan|
sn sna Shona|so som Somali|sq sqi Albanian|sr srp Serbian|ss ssw Swati|st sot Sotho|su sun Sundanese|
sv swe Swedish|sw swa Swahili|ta tam Tamil|te tel Telugu|tg tgk Tajik|th tha Thai|ti tir Tigrinya|
tk tuk Turkmen|tl tgl Tagalog|tn tsn Tswana|to ton Tongan|tr tur Turkish|ts tso Tsonga|tt tat Tatar|
tw twi Twi|ty tah Tahitian|ug uig Uyghur|uk ukr Ukrainian|ur urd Urdu|uz uzb Uzbek|ve ven Venda|
vi vie Vietnamese|vo vol Volapuk|wa wln Walloon|wo wol Wolof|xh xho Xhosa|yi yid Yiddish|
yo yor Yoruba|za zha Zhuang|zh zho Chinese|zu zul Zulu
"""

# ISO 639-2/B (bibliographic) codes that differ from 639-3.
_ISO2B = {
    "alb": "sqi", "arm": "hye", "baq": "eus", "bur": "mya", "chi": "zho", "cze": "ces",
    "dut": "nld", "fre": "fra", "geo": "kat", "ger": "deu", "gre": "ell", "ice": "isl",
    "mac": "mkd", "mao": "mri", "may": "msa", "per": "fas", "rum": "ron", "slo": "slk",
    "tib": "bod", "wel": "cym",
}

# Macrolanguages (and Serbo-Croatian) -> member languages, most common first.
MACRO_MEMBERS: Dict[str, List[str]] = {
    "zho": ["cmn", "yue", "nan", "wuu", "hak", "cdo", "cjy", "hsn", "gan", "cpx", "czh", "mnp"],
    "ara": ["arb", "arz", "apc", "ajp", "ary", "acm", "ayl", "afb", "acw", "aeb", "arq"],
    "fas": ["pes", "prs"], "msa": ["zsm", "ind"], "nor": ["nob", "nno"],
    "sqi": ["als", "aln"], "est": ["ekk"], "lav": ["lvs"], "aze": ["azj", "azb"],
    "uzb": ["uzn"], "kur": ["kmr", "ckb", "sdh"], "swa": ["swh"], "grn": ["gug"],
    "hbs": ["hrv", "srp", "bos"], "mon": ["khk"], "ori": ["ory"], "pus": ["pbu", "pbt"],
    "yid": ["ydd"], "mlg": ["plt"], "que": ["quz", "qub"], "zza": ["diq", "kiu"],
}

ISO1_TO_3: Dict[str, str] = {}
NAMES: Dict[str, str] = {}  # lowercase English name -> 639-3
CODE_NAMES: Dict[str, str] = {}  # 639-3 -> English name
for item in _ISO1.replace("\n", "").split("|"):
    two, three, name = item.strip().split(" ", 2)
    ISO1_TO_3[two] = three
    NAMES[name.lower()] = three
    CODE_NAMES[three] = name
NAMES.update({"mandarin": "cmn", "cantonese": "yue", "hokkien": "nan", "min nan": "nan",
              "farsi": "fas", "slovene": "slv", "flemish": "nld", "castilian": "spa",
              "filipino": "tgl", "kurmanji": "kmr", "sorani": "ckb", "old english": "ang",
              "ancient greek": "grc", "serbo-croatian": "hbs", "upper sorbian": "hsb"})
CODE_NAMES.update({"cmn": "Mandarin", "yue": "Cantonese", "nan": "Min Nan", "hbs": "Serbo-Croatian",
                   "ang": "Old English", "grc": "Ancient Greek", "kmr": "Kurmanji", "ckb": "Sorani",
                   "hsb": "Upper Sorbian", "gug": "Paraguayan Guarani", "enm": "Middle English"})


def normalize_code(code: str) -> str:
    """'DE' -> 'deu', 'en_US' -> 'eng-us', 'German' -> 'deu', 'pt-BR' -> 'por-br'."""
    raw = code.strip()
    low = raw.lower().replace("_", "-")
    if low in NAMES:
        return NAMES[low]
    spaced = low.replace("-", " ")
    if spaced in NAMES:
        return NAMES[spaced]
    base, _, variety = low.partition("-")
    base = ISO1_TO_3.get(base, _ISO2B.get(base, base))
    if base in NAMES:  # e.g. "english-us"
        base = NAMES[base]
    return f"{base}-{variety}" if variety else base


def name_of(tag: str) -> str:
    base, _, variety = tag.partition("-")
    name = CODE_NAMES.get(base, base)
    return f"{name} ({variety})" if variety else name
