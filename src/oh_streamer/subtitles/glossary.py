"""Versioned Sakura glossary and official prompt construction."""
from dataclasses import dataclass
import json
from pathlib import Path

SYSTEM = (
    '你是一个轻小说翻译模型，可以流畅通顺地以日本轻小说的风格将日文翻译成简体中文，'
    '并联系上下文正确使用人称代词，不擅自添加原文中没有的代词。'
)
PREFIX = '根据以下术语表（可以为空）：\n'
GALTRANSL_SYSTEM = (
    '你是一个视觉小说翻译模型，可以通顺地使用给定的术语表以指定的风格将日文翻译成简体中文，'
    '并联系上下文正确使用人称代词，注意不要混淆使役态和被动态的主语和宾语，'
    '不要擅自添加原文中没有的特殊符号，也不要擅自增加或减少换行。'
)
GALTRANSL_PREFIX = '参考以下术语表（可为空，格式为src->dst #备注）：\n'
DEFAULT_TRANSLATE_MODEL = 'sakura-v1'


@dataclass(frozen=True)
class SakuraProfile:
    name: str
    system: str
    prefix: str
    temperature: float
    top_p: float


SAKURA_PROFILES = {
    DEFAULT_TRANSLATE_MODEL: SakuraProfile(DEFAULT_TRANSLATE_MODEL, SYSTEM, PREFIX, .1, .3),
    **{
        name: SakuraProfile(name, GALTRANSL_SYSTEM, GALTRANSL_PREFIX, .3, .8)
        for name in ('sakura-galtransl-7b-v3.7', 'sakura-galtransl-14b-v3.8')
    },
}


def get_sakura_profile(name=DEFAULT_TRANSLATE_MODEL):
    name = str(name).strip().lower()
    name = {'v1': DEFAULT_TRANSLATE_MODEL, 'v1.0': DEFAULT_TRANSLATE_MODEL}.get(name, name)
    try:
        return SAKURA_PROFILES[name]
    except KeyError:
        raise ValueError('unknown Sakura translation model: ' + name) from None


def load_glossary(path):
    if path is None:
        return []
    path = Path(path)
    if path.suffix.lower() == '.json':
        data = json.loads(path.read_text('utf-8-sig'))
        if isinstance(data, dict):
            data = [dict(src=k, dst=v) for k, v in data.items()]
        if not isinstance(data, list):
            raise ValueError('glossary JSON must be a list or mapping')
    else:
        data = []
        for line in path.read_text('utf-8-sig').splitlines():
            line = line.strip()
            if not line or line.startswith('#'):
                continue
            entry, _, info = line.partition('#')
            src, sep, dst = entry.partition('->')
            if not sep:
                raise ValueError('glossary line requires src->dst #note')
            data.append(dict(src=src.strip(), dst=dst.strip(), info=info.strip()))
    for item in data:
        if not isinstance(item, dict) or not all(
            isinstance(item.get(k), str) and item[k].strip()
            for k in ('src', 'dst')
        ):
            raise ValueError('glossary entries require nonempty src and dst strings')
        if any(
            '\n' in str(item.get(k, '')) or '\r' in str(item.get(k, ''))
            for k in ('src', 'dst', 'info')
        ):
            raise ValueError('glossary fields must be single-line')
    return data


def sakura_messages(japanese, glossary=(), context='', *, translate_model=DEFAULT_TRANSLATE_MODEL):
    profile = get_sakura_profile(translate_model)
    terms = '\n'.join(
        r['src'] + '->' + r['dst'] + (' #' + r['info'] if r.get('info') else '')
        for r in glossary
    )
    if profile.name == DEFAULT_TRANSLATE_MODEL:
        content = PREFIX + terms + '\n将下面的日文文本根据对应关系和备注翻译成中文：' + japanese
        if context:
            content = context + '\n\n' + content
    else:
        content = (
            ('历史翻译：' + context + '\n' if context else '')
            + profile.prefix + terms
            + '\n根据以上术语表的对应关系和备注，结合历史剧情和上下文，将下面的文本从日文翻译成简体中文：\n'
            + japanese
        )
    return [
        {'role': 'system', 'content': profile.system},
        {'role': 'user', 'content': content},
    ]


def parse_lines(text, ids):
    rows = [r.strip() for r in text.splitlines() if r.strip()]
    if len(rows) != len(ids):
        raise ValueError('translation line count mismatch')
    return '\n'.join(f'{i}. {text}' for i, text in zip(ids, rows))


def general_llm_messages(japanese, glossary=(), context=''):
    """Build messages for general-purpose LLMs translating to Traditional Chinese (Taiwan)."""
    terms = '\n'.join(
        r['src'] + ' -> ' + r['dst'] + (' #' + r['info'] if r.get('info') else '')
        for r in glossary
    )
    sys_parts = [
        '你是一個專業的影視字幕翻譯與審校模型。請將給定的日文對白逐句翻譯成流暢自然的繁體中文（台灣習慣用詞）。',
        '【重要過濾與情節台詞保護規則】：',
        '1. 【純狀聲詞與幻聽過濾】：純粹為孤立歡呼、無語意笑聲、親吻吸吮、孤立喘息或呻吟（如：うぇーい、ハハハハハ、ちゅっ、ふふっ、ああ、ハァハァ等）以及 ASR 宣傳幻聽（ご視聴ありがとうございました、チャンネル登録等），請直接輸出「[SKIP]」，不要輸出中文字幕。',
        '2. 【身材特徵與親熱暗語保護】：形容角色身材特徵（如「ボン、キュッ、ボン！」前凸後翹、「プルンプルン」Ｑ彈Ｑ彈）與親熱暗語（「チョメチョメ」嘿咻嘿咻/啪啪啪）為關鍵情節台詞，明訂嚴禁跳過，絕對不可輸出 [SKIP]，必須生動翻譯。',
        '3. 【混合行清理保留對話】：若句子開頭或結尾帶有狀聲詞（如「ぢゅぱっ、やめれない…」），請去除句首句尾狀聲詞，保留並翻譯實際對話台詞（如「停不下來……」）。',
        '4. 嚴格保留每一行的原始編號（格式為「數字. 譯文」或「數字. [SKIP]」），一句一行，不得遺漏或合併任何編號。',
        '5. 僅輸出翻譯結果或 [SKIP]，絕對不要輸出任何解釋、開場白、備註、markdown 程式碼標記或上下文說明。',
        '6. 翻譯須符合台灣口語與在地習慣用詞，不得保留任何日文假名。',
    ]
    if terms:
        sys_parts.append('7. 參考以下術語表對照進行翻譯：\n' + terms)
    system_content = '\n'.join(sys_parts)

    user_parts = []
    if context:
        user_parts.append('【前文參考（僅供理解上下文，請勿重複翻譯）】\n' + context)
    user_parts.append('【請將以下日文依編號逐句翻譯成繁體中文（台灣）】\n' + japanese)
    return [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': '\n\n'.join(user_parts)},
    ]


def general_llm_single_messages(text, glossary=()):
    """Build messages for single-cue general LLM translation."""
    terms = '\n'.join(
        r['src'] + ' -> ' + r['dst'] + (' #' + r['info'] if r.get('info') else '')
        for r in glossary
    )
    sys_parts = [
        '你是一個專業的影視字幕翻譯與審校模型。請將給定的日文句子翻譯成流暢自然的繁體中文（台灣習慣用詞）。',
        '【重要過濾與情節台詞保護規則】：',
        '1. 【純狀聲詞與幻聽過濾】：若純粹為孤立歡呼、無語意笑聲、親吻吸吮、孤立喘息或呻吟，或 ASR 幻聽（ご視聴ありがとうございました等），請直接輸出「1. [SKIP]」。',
        '2. 【身材特徵與親熱暗語保護】：形容角色身材特徵（如「ボン、キュッ、ボン！」、「プルンプルン」）與親熱暗語（「チョメチョメ」）為關鍵情節台詞，嚴禁跳過，必須生動翻譯。',
        '3. 【混合行清理保留對話】：去除句首句尾狀聲詞，保留實際對白台詞。',
        '4. 請以「1. 譯文」或「1. [SKIP]」的格式輸出翻譯結果。',
        '5. 僅輸出該行翻譯或 [SKIP]，絕對不要輸出任何解釋、備註或多餘文字。',
        '6. 翻譯須符合台灣口語習慣，不得保留任何日文假名。',
    ]
    if terms:
        sys_parts.append('7. 參考以下術語表對照進行翻譯：\n' + terms)
    system_content = '\n'.join(sys_parts)
    return [
        {'role': 'system', 'content': system_content},
        {'role': 'user', 'content': '1. ' + text},
    ]


def filter_relevant_terms(glossary=(), text=''):
    """Filter glossary terms to only those whose source Japanese term appears in text.

    This prevents hitting cloud token-per-minute (TPM) limits and keeps LLM context clean.
    """
    if not glossary or not text:
        return []
    clean_text = str(text)
    return [
        term for term in glossary
        if isinstance(term, dict) and term.get('src') and str(term['src']) in clean_text
    ]

