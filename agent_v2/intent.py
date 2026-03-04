"""Rule-based intent classification and parameter extraction."""
from __future__ import annotations

import re
from enum import Enum
from typing import Any


class Intent(Enum):
    GREETING = "greeting"
    CAPABILITY = "capability"
    THANKS = "thanks"
    BYE = "bye"
    SEARCH = "search"
    FILTER = "filter"
    RENT = "rent"
    TERMINATE = "terminate"
    COMPARE_PRICE = "compare_price"
    NEARBY_LANDMARK = "nearby_landmark"
    COMMUNITY_QUERY = "community_query"
    LANDMARK_SEARCH = "landmark_search"
    COMPLEX = "complex"


# ── Constants ──────────────────────────────────────────────────────────

DISTRICTS = ["海淀", "朝阳", "通州", "昌平", "大兴", "房山", "西城", "丰台", "顺义", "东城"]

_CN_NUM = {"一": "1", "二": "2", "两": "2", "三": "3", "四": "4", "五": "5"}

PLATFORMS = {"链家": "链家", "安居客": "安居客", "58同城": "58同城", "58": "58同城"}

LANDMARK_KEYWORDS = {
    "望京南": ("朝阳", "望京南"),
    "望京西": ("朝阳", "望京西"),
    "望京": ("朝阳", "望京"),
    "立水桥": ("朝阳", "立水桥"),
    "双合站": ("朝阳", "双合"),
    "百子湾": ("朝阳", "百子湾"),
    "三元桥": ("朝阳", "三元桥"),
    "金融街": ("西城", "金融街"),
    "西二旗": ("海淀", "西二旗"),
    "车公庄": ("西城", "车公庄"),
    "中关村": ("海淀", "中关村"),
    "国贸": ("朝阳", "国贸"),
    "上地": ("海淀", "上地"),
    "亦庄": ("大兴", "亦庄"),
    "房山城关": ("房山", "房山城关"),
}

# ── Regex patterns ─────────────────────────────────────────────────────

_BEDROOM_RE = re.compile(r"([一二两三四五1-5])\s*(?:居|室|房)")
_PRICE_RANGE_RE = re.compile(r"(\d+(?:k|千)?)\s*(?:到|至|-|~)\s*(\d+(?:k|千)?)", re.I)
_PRICE_MAX_RE = re.compile(r"(?:租金|预算|月租|价格|房租).*?(\d+(?:k|千)?)\s*(?:元|块)?(?:以[下内里]|以内|之内)?", re.I)
_PRICE_MAX_RE2 = re.compile(r"(\d+(?:k|千)?)\s*(?:元|块)?(?:以[下内里]|以内|之内)", re.I)
_PRICE_AROUND_RE = re.compile(r"(?:预算|月租|价格|房租).*?(\d{3,5})\s*(?:左右|上下)", re.I)
_AREA_MIN_RE = re.compile(r"(\d{2,3})\s*(?:平|㎡)(?:以上|米以上)?")
_COMMUTE_RE = re.compile(r"通勤\s*(\d{1,3})\s*分钟")
_SUBWAY_LINE_RE = re.compile(r"(\d{1,2})号线")
_SUBWAY_DIST_RE = re.compile(r"(\d{3,4})\s*(?:米|m)(?:以?内)?")
_COMMUNITY_RE = re.compile(r"^(.+?)(?:有)?在租", re.I)
_LANDMARK_DIST_RE = re.compile(r"(\d{3,4})\s*米")
_AVAILABLE_DATE_RE = re.compile(r"(\d{1,2})月(\d{1,2})(?:号|日)")

# ── Tag mappings ───────────────────────────────────────────────────────
# keyword -> (require_tags, exclude_tags, field_filters)
# field_filters: list of (field, op, value) tuples

TAG_RULES: list[tuple[list[str], list[str], list[str], list[tuple[str, str, Any]]]] = [
    # (keywords, require_tags, exclude_tags, field_filters)
    # -- 宠物 --
    (["养金毛", "养大型犬", "养哈士奇", "养德牧", "养拉布拉多", "养阿拉斯加"],
     ["可养狗"], ["仅限小型犬", "不可养宠物"], []),
    (["养小型犬", "养小狗"], ["仅限小型犬"], ["不可养宠物"], []),
    (["可养狗", "养狗", "能养狗", "允许养狗", "需要房东允许养狗"],
     ["可养狗"], ["不可养宠物"], []),
    (["可养猫", "养猫", "能养猫", "允许养猫", "接受养猫", "能接受养猫", "养了一只猫", "养了只猫",
      "养了只英短", "养了只布偶猫", "养了只柯基", "房东允许养猫"],
     ["可养猫"], ["不可养宠物"], []),
    (["养仓鼠", "仓鼠", "可养宠物", "能养宠物", "接受宠物", "能接受宠物", "允许养宠物", "养宠物"],
     ["可养宠物"], ["不可养宠物"], []),
    (["不养宠物", "室友不养", "对宠物过敏", "不要养宠物", "不养狗", "不养猫", "室友不养宠物"],
     [], ["可养狗", "可养猫", "可养宠物"], []),
    (["不额外收宠物押金", "不要宠物押金", "免宠物押金", "不额外收宠物", "不交宠物押金"],
     [], ["可养宠物需宠物押金"], []),
    (["接受交宠物押金", "可以交宠物押金", "接受宠物押金"], [], [], []),
    # -- 安静/噪音 --
    (["安静", "隔音", "睡眠浅", "怕吵", "不能吵", "不隔音", "睡眠不好", "隔音要好", "休养", "静养"],
     [], [], [("hidden_noise_level", "not_in", ["吵闹", "临街"])]),
    # -- 朝向 --
    (["朝南"], [], [], [("orientation", "contains", "朝南")]),
    (["南北通透", "南北"], [], [], [("orientation", "contains", "南北")]),
    # -- 电梯 --
    (["电梯", "有电梯", "带电梯", "不想爬楼", "爬楼费劲", "腿脚不便", "腿脚不好", "腿脚不太好",
      "腿脚不太方便", "不想走楼梯", "不能爬楼"],
     [], [], [("elevator", "eq", True)]),
    # -- 民水民电 --
    (["民水民电", "民电"], [], [], [("utilities_type", "contains", "民水民电")]),
    # -- 费用 --
    (["包水电费", "包水电", "水电费包在房租", "水电费包", "租金包水电", "水电包在房租里", "租金包水电费", "包水电费"],
     ["包水电费"], ["水电费另付"], []),
    (["免水电费", "免水电"], ["免水电费"], ["水电费另付"], []),
    (["包宽带", "网费包在房租", "网费包", "宽带包", "网费包含", "包网", "不想交网费",
      "不想额外再交网费", "宽带能好", "宽带包在房租里", "包含在房租里"],
     ["包宽带"], ["网费另付"], []),
    (["免宽带费"], ["免宽带费"], ["网费另付"], []),
    (["包物业费", "物业费包在房租", "物业费包", "物业费含在房租", "物业费能包"],
     ["包物业费"], ["物业费另付"], []),
    (["免物业费"], ["免物业费"], ["物业费另付"], []),
    (["免车位费", "车位免费", "免费车位", "包车位", "包车位费", "包含车位费"],
     ["免车位费"], ["车位费另付"], []),
    (["车库车位", "地下车库", "有车库", "地库车位"],
     ["车库车位"], [], []),
    (["有车位", "有车", "带车位", "小区有车位", "需要车位"], [], ["无车位"], []),
    # -- 押付 --
    (["月付", "按月付", "每月一付", "按月支付"], ["月付"], [], []),
    (["押一付一", "押一"], ["押一"], [], []),
    (["押二付一", "押二", "两个月押金", "2个月押金", "接受押二", "可以接受押二", "接受2个月的押金",
      "可以接受2个月的押金", "接受两个月"],
     ["押二"], [], []),
    (["季付"], ["季付"], [], []),
    (["半年付"], ["半年付"], [], []),
    (["年付", "接受年付"], ["年付"], [], []),
    # -- 租期 --
    (["可月租", "短租"], ["可月租"], ["仅接受年租"], []),
    (["租2个月", "租两个月", "可租2个月", "只住一个多月"], ["可租2个月"], ["仅接受年租"], []),
    (["租3个月", "最多租三个月", "可租3个月", "租个几个月", "只租两个月"],
     ["可租3个月"], ["仅接受年租"], []),
    (["长租", "签半年", "半年以上"], ["可半年租"], [], []),
    # -- 中介 --
    (["房东直租", "无中介", "不想通过中介", "直接跟房东", "不想交中介费", "免中介费",
      "不交中介费", "省中介费", "不想找中介", "省去中介费"],
     ["房东直租"], ["收中介费"], []),
    (["中介费一月租", "中介费按一个月", "中介费一个月"], ["中介费一月租"], [], []),
    (["中介费半月租", "中介费按半个月"], ["中介费半月租"], [], []),
    # -- 看房方式 --
    (["线上VR看房", "VR看房", "线上VR"], ["仅线上VR看房"], ["仅线下看房"], []),
    (["线上看房", "不用跑现场"], [], ["仅线下看房"], []),
    (["线上图片看房", "线上图片"], ["仅线上图片看房"], ["仅线下看房"], []),
    (["线上AR看房", "线上AR"], ["仅线上AR看房"], ["仅线下看房"], []),
    (["实地看房", "线下看房", "去实地看房", "去看房"],
     [], ["仅线上VR看房", "仅线上图片看房", "仅线上AR看房"], []),
    (["周末看房", "只能周末看房", "周末有空", "只有周末", "仅周六周日", "周六日有空",
      "只有周末方便", "只有周末有时间"],
     [], ["仅工作日看房"], []),
    (["工作日看房", "工作日能看房", "工作日白天看房", "工作日白天和下午",
      "工作日14-18点看房", "工作日14-18点"],
     [], ["仅周末看房"], []),
    (["下午看房", "下午能看房"], [], [], []),
    (["全天可看房", "时间灵活", "随时可以看房", "全天能约"], ["全天可看房"], [], []),
    # -- 周边设施 --
    (["近公园", "附近有公园", "公园", "遛狗", "附近有个公园", "散步", "跑步", "户外跑步", "写生", "遛弯"],
     ["近公园"], [], []),
    (["近菜市场", "附近有菜市场", "菜市场", "买菜方便", "买菜"],
     ["近菜市场"], [], []),
    (["近医院", "附近有医院", "医院", "离医院近", "复查方便"],
     ["近医院"], [], []),
    (["近学校", "附近有学校", "学校", "离学校近"],
     ["近学校"], [], []),
    (["近健身房", "附近有健身房", "健身房", "健身", "锻炼", "力量训练", "24小时健身房"],
     ["近健身房"], [], []),
    (["近商超", "附近有商场", "商场", "逛街", "超市", "附近有商超", "便利店", "商场超市"],
     ["近商超"], [], []),
    (["近餐饮", "附近有餐馆", "餐馆", "餐饮", "24小时有吃的", "24小时有餐饮",
      "小吃街", "吃饭的地方", "吃饭方便", "解决吃饭", "有吃的", "网红餐饮"],
     ["近餐饮"], [], []),
    (["近警察局", "附近有派出所", "派出所", "警察局"],
     ["近警察局"], [], []),
    (["近银行", "附近有银行"], ["近银行"], [], []),
    (["近加油站", "附近有加油站", "加油站", "加油方便"],
     ["近加油站"], [], []),
    # -- 其他 --
    (["24小时保安", "保安", "晚归安全"],
     ["24小时保安"], [], []),
    (["门禁刷卡", "刷卡进", "门禁"],
     ["门禁刷卡"], [], []),
    (["采光好", "采光", "有阳光", "明亮", "亮堂", "光线好"],
     ["采光好"], [], []),
    (["房东好沟通", "好沟通的房东", "房东好说话"],
     ["房东好沟通"], [], []),
    (["提前退租可协商", "协商退租", "可以协商退租", "提前结束"],
     ["提前退租可协商"], [], []),
    (["绿化好", "绿化好环境佳", "环境好", "绿地"],
     ["绿化好环境佳"], [], []),
    (["高性价比", "性价比高"], ["高性价比"], [], []),
    (["物业管理到位", "物业管理好", "物业好"], ["物业管理到位"], [], []),
    (["合同规范", "合同清晰"], ["合同规范条款清晰"], [], []),
]


# ── Price parser ───────────────────────────────────────────────────────

def _parse_price(s: str) -> int:
    s = s.lower().strip()
    if "k" in s:
        return int(float(s.replace("k", "")) * 1000)
    if "千" in s:
        return int(float(s.replace("千", "")) * 1000)
    return int(s)


# ── Main extraction ────────────────────────────────────────────────────

def classify_intent(msg: str, has_candidates: bool, turn: int) -> Intent:
    """Classify user message intent."""
    m = msg.strip().lower()

    if m in ("你好", "hello", "hi", "嗨", "您好", "你好呀", "hi~", "hello~"):
        return Intent.GREETING
    cap_kws = ["你可以做什么", "你是谁", "你能做什么", "你的功能", "你能帮我做什么", "你会什么"]
    if any(x in m for x in cap_kws):
        return Intent.CAPABILITY
    if m in ("谢谢", "感谢", "多谢", "谢了", "thanks", "thank you", "好的谢谢"):
        return Intent.THANKS
    if m in ("再见", "拜拜", "bye", "结束"):
        return Intent.BYE

    # Terminate
    if any(p in msg for p in ("退租", "退掉", "我要退", "帮我退租", "帮我退掉")):
        return Intent.TERMINATE

    # Rent intent
    rent_pats = [
        "就租第一套", "就定第一套", "就选第一套", "就第一套", "我要租这套",
        "就选最便宜的那个平台", "就选最便宜的平台",
        "就租第一套吧", "就定第一套吧", "就选第一套吧",
        "就第一套吧", "我们租", "我们要租", "我要租",
    ]
    if any(p in msg for p in rent_pats):
        return Intent.RENT

    # Compare price
    if ("挂牌价" in msg or "哪个最便宜" in msg) and ("第一套" in msg or "这套" in msg):
        return Intent.COMPARE_PRICE

    # Nearby landmark query (附近有X吗?)
    lm_types = ["公园", "菜市场", "医院", "学校", "健身房", "商超", "商场", "餐饮", "餐馆", "警察局", "派出所"]
    if has_candidates and any(t in msg for t in lm_types):
        if any(p in msg for p in ("附近", "周边")) and any(p in msg for p in ("吗", "有没有", "有吗", "请问")):
            return Intent.NEARBY_LANDMARK

    # Community query
    if "在租" in msg:
        cm = _COMMUNITY_RE.search(msg)
        if cm:
            community = cm.group(1).strip()
            if len(community) >= 2 and not any(d in community for d in DISTRICTS):
                return Intent.COMMUNITY_QUERY

    # If we have candidates and the message looks like a filter (tag/attribute check)
    if has_candidates and turn > 1:
        filter_kws = _extract_tag_specs(msg)
        if filter_kws["tags_require"] or filter_kws["tags_exclude"] or filter_kws["field_filters"]:
            return Intent.FILTER

    # Landmark search (mentions a known landmark for nearby search)
    for lm in LANDMARK_KEYWORDS:
        if lm in msg and any(kw in msg for kw in ["附近", "站", "那边", "这边", "周边"]):
            return Intent.LANDMARK_SEARCH

    # General search intent
    search_kws = ["找", "租", "房", "居室", "居", "套", "单间", "推荐", "看看", "希望", "想要", "预算", "有没有", "有吗"]
    if any(kw in msg for kw in search_kws):
        # If we already have candidates and this looks like adding conditions -> FILTER
        if has_candidates and turn > 1:
            new_params = extract_search_params(msg)
            # If the only new thing is params we already have, treat as filter
            has_new_district = new_params.get("district") is not None
            if not has_new_district:
                filter_kws = _extract_tag_specs(msg)
                if filter_kws["tags_require"] or filter_kws["tags_exclude"] or filter_kws["field_filters"]:
                    return Intent.FILTER
                # Check if user is changing price/bedrooms etc. -> new search
                if any(new_params.get(k) for k in ("max_price", "min_price", "bedrooms", "decoration", "max_subway_dist")):
                    return Intent.FILTER
        return Intent.SEARCH

    # Default: if has candidates and message could be a filter
    if has_candidates:
        filter_kws = _extract_tag_specs(msg)
        if filter_kws["tags_require"] or filter_kws["tags_exclude"] or filter_kws["field_filters"]:
            return Intent.FILTER

    return Intent.COMPLEX


def extract_search_params(msg: str) -> dict[str, Any]:
    """Extract structured search parameters from user message."""
    params: dict[str, Any] = {}

    # District
    for d in DISTRICTS:
        if d in msg:
            params["district"] = d
            break
    if not params.get("district"):
        for lm, (dist, _) in LANDMARK_KEYWORDS.items():
            if lm in msg:
                params["district"] = dist
                break

    # Bedrooms
    bed_m = _BEDROOM_RE.search(msg)
    if bed_m:
        raw = bed_m.group(1)
        params["bedrooms"] = _CN_NUM.get(raw, raw)
    elif "单间" in msg:
        params["rental_type"] = "合租"

    # Price
    range_m = _PRICE_RANGE_RE.search(msg)
    if range_m:
        p1 = _parse_price(range_m.group(1))
        p2 = _parse_price(range_m.group(2))
        params["min_price"] = min(p1, p2)
        params["max_price"] = max(p1, p2)
    else:
        around_m = _PRICE_AROUND_RE.search(msg)
        if around_m:
            base = int(around_m.group(1))
            params["min_price"] = int(base * 0.7)
            params["max_price"] = int(base * 1.3)
        else:
            price_m = _PRICE_MAX_RE.search(msg) or _PRICE_MAX_RE2.search(msg)
            if price_m:
                params["max_price"] = _parse_price(price_m.group(1))

    # Area
    area_m = _AREA_MIN_RE.search(msg)
    if area_m:
        params["min_area"] = int(area_m.group(1))

    # Decoration
    if "空房" in msg or "毛坯" in msg:
        params["decoration"] = "毛坯"
    elif "简装" in msg:
        params["decoration"] = "简装"
    elif "精装" in msg:
        params["decoration"] = "精装"
    elif "豪装" in msg or "豪华" in msg:
        params["decoration"] = "豪华"

    # Elevator
    if any(kw in msg for kw in ["电梯", "不想爬楼", "爬楼费劲", "腿脚不便", "腿脚不好",
                                 "腿脚不太好", "腿脚不太方便", "不想走楼梯", "不能爬楼"]):
        params["elevator"] = "true"

    # Subway distance
    if "近地铁" in msg:
        params["max_subway_dist"] = 800
    elif "地铁可达" in msg:
        params["max_subway_dist"] = 1000
    else:
        sd_m = _SUBWAY_DIST_RE.search(msg)
        if sd_m and ("地铁" in msg or "离地铁" in msg):
            params["max_subway_dist"] = int(sd_m.group(1))
        elif "离地铁近" in msg or "地铁近" in msg:
            params["max_subway_dist"] = 800
    # "一公里以内" / "一千米"
    if "一公里" in msg and ("地铁" in msg or "离地铁" in msg):
        params["max_subway_dist"] = 1000
    if "两公里" in msg and ("地铁" in msg or "离地铁" in msg):
        params["max_subway_dist"] = 2000
    if "500米" in msg and ("地铁" in msg or "离地铁" in msg):
        params["max_subway_dist"] = 500
    if "走路十分钟" in msg or "走路10分钟" in msg:
        params["max_subway_dist"] = 800

    # Commute
    commute_m = _COMMUTE_RE.search(msg)
    if commute_m:
        params["commute_to_xierqi_max"] = int(commute_m.group(1))

    # Rental type
    if "整租" in msg or "自己住" in msg or "一个人住" in msg or "不跟别人合租" in msg:
        params["rental_type"] = "整租"
    elif "合租" in msg or "单间" in msg or "有个室友" in msg or "找人合租" in msg:
        params["rental_type"] = "合租"

    # Platform
    for label, plat in PLATFORMS.items():
        if label in msg:
            params["listing_platform"] = plat
            break

    # Sort
    if "从低到高" in msg or "从便宜到贵" in msg or "按便宜到贵" in msg or "按租金从小到大" in msg:
        params["sort_by"] = "price"
        params["sort_order"] = "asc"
    elif "从高到低" in msg or "从贵到便宜" in msg:
        params["sort_by"] = "price"
        params["sort_order"] = "desc"
    elif "从大到小" in msg:
        params["sort_by"] = "area"
        params["sort_order"] = "desc"
    elif "从小到大" in msg:
        params["sort_by"] = "area"
        params["sort_order"] = "asc"
    elif "离地铁从近到远" in msg or "按离地铁从近到远" in msg:
        params["sort_by"] = "subway"
        params["sort_order"] = "asc"

    # Subway line
    sl_m = _SUBWAY_LINE_RE.search(msg)
    if sl_m:
        params["subway_line"] = sl_m.group(1)

    # Available date
    date_m = _AVAILABLE_DATE_RE.search(msg)
    if date_m:
        month = int(date_m.group(1))
        day = int(date_m.group(2))
        params["available_from_before"] = f"2026-{month:02d}-{day:02d}"

    # Orientation
    if "南北通透" in msg:
        params["orientation"] = "南北"
    elif "朝南" in msg:
        params["orientation"] = "朝南"

    return params


def _extract_tag_specs(msg: str) -> dict[str, list]:
    """Extract tag requirements and exclusions from message."""
    require: list[str] = []
    exclude: list[str] = []
    field_filters: list[tuple[str, str, Any]] = []

    for keywords, req_tags, exc_tags, ff in TAG_RULES:
        if any(kw in msg for kw in keywords):
            for t in req_tags:
                if t not in require:
                    require.append(t)
            for t in exc_tags:
                if t not in exclude:
                    exclude.append(t)
            for f in ff:
                if f not in field_filters:
                    field_filters.append(f)

    return {"tags_require": require, "tags_exclude": exclude, "field_filters": field_filters}


def extract_filter_specs(msg: str) -> dict[str, Any]:
    """Extract filter specifications including tags and field checks.
    Also extracts any updated search params (price changes, etc.)."""
    specs = _extract_tag_specs(msg)
    # Also check for price/bedrooms changes
    params = extract_search_params(msg)
    specs["updated_params"] = params
    return specs


def extract_community_name(msg: str) -> str | None:
    """Extract community name from '建清园南区有在租的吗' pattern."""
    if "在租" not in msg:
        return None
    m = _COMMUNITY_RE.search(msg)
    if not m:
        return None
    community = m.group(1).strip()
    if len(community) < 2 or any(d in community for d in DISTRICTS):
        return None
    for suffix in ["南区", "北区", "东区", "西区", "一期", "二期"]:
        if community.endswith(suffix) and "(" not in community:
            community = community[:-len(suffix)] + f"({suffix})"
            break
    return community


def extract_landmark_query(msg: str) -> tuple[str, int] | None:
    """Extract landmark name and max_distance for nearby search."""
    for lm in LANDMARK_KEYWORDS:
        if lm in msg:
            dist_m = _LANDMARK_DIST_RE.search(msg)
            max_dist = int(dist_m.group(1)) if dist_m else 2000
            if "500米" in msg:
                max_dist = 500
            return (lm, max_dist)
    return None


def get_canned_response(intent: Intent) -> str | None:
    """Return canned response for simple intents."""
    if intent == Intent.GREETING:
        return "您好！我是智能租房助手，请问有什么可以帮您？"
    if intent == Intent.CAPABILITY:
        return "您好！我是智能租房助手，可以帮您找房、查房、租房、退租，以及查询房源周边配套设施。"
    if intent == Intent.THANKS:
        return "不客气，如有其他需求随时联系我！"
    if intent == Intent.BYE:
        return "再见，祝您找到满意的房子！"
    return None
