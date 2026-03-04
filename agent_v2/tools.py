"""OpenAI-format tool definitions for the rental API."""


def get_tools_schema(round_num: int = 1) -> list[dict]:
    """Return tool definitions. round_num>=2 includes rent/terminate/offline."""

    search_tools = [
        {
            "type": "function",
            "function": {
                "name": "search_landmarks",
                "description": "关键词模糊搜索地标（地铁站/公司/商圈），返回landmark_id用于get_houses_nearby查附近房源。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "q": {"type": "string", "description": "搜索关键词，如「望京南」「百度」「西二旗」"},
                        "category": {"type": "string", "description": "类别：subway/company/landmark，可选"},
                        "district": {"type": "string", "description": "行政区，可选"},
                    },
                    "required": ["q"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_houses_nearby",
                "description": "以地标为圆心查附近可租房源。需先用search_landmarks获取landmark_id。返回带距离、步行时间。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "landmark_id": {"type": "string", "description": "地标ID，从search_landmarks结果获取"},
                        "max_distance": {"type": "number", "description": "最大直线距离(米)，默认2000"},
                        "page_size": {"type": "integer", "description": "每页条数，默认10，建议50"},
                    },
                    "required": ["landmark_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_houses_by_platform",
                "description": "条件查房：按区域/价格/户型/装修/电梯/地铁距离/朝向等筛选可租房源。地标附近查房请用search_landmarks+get_houses_nearby。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "district": {"type": "string", "description": "行政区：海淀/朝阳/通州/昌平/大兴/房山/西城/丰台/顺义/东城"},
                        "area": {"type": "string", "description": "商圈，如西二旗/上地/望京"},
                        "min_price": {"type": "integer", "description": "最低月租(元)"},
                        "max_price": {"type": "integer", "description": "最高月租(元)"},
                        "bedrooms": {"type": "string", "description": "卧室数，如「2」「1,2」"},
                        "rental_type": {"type": "string", "description": "整租或合租"},
                        "decoration": {"type": "string", "description": "精装/简装/豪华/毛坯/空房"},
                        "orientation": {"type": "string", "description": "朝向：朝南/朝北/南北等"},
                        "elevator": {"type": "string", "description": "是否有电梯：true/false"},
                        "min_area": {"type": "integer", "description": "最小面积(㎡)"},
                        "max_area": {"type": "integer", "description": "最大面积(㎡)"},
                        "max_subway_dist": {"type": "integer", "description": "最大地铁距离(米)，近地铁=800"},
                        "subway_station": {"type": "string", "description": "地铁站名"},
                        "subway_line": {"type": "string", "description": "地铁线路，如13"},
                        "commute_to_xierqi_max": {"type": "integer", "description": "到西二旗通勤上限(分钟)"},
                        "utilities_type": {"type": "string", "description": "民水民电/商水商电"},
                        "available_from_before": {"type": "string", "description": "可入住日期上限，YYYY-MM-DD"},
                        "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]},
                        "sort_by": {"type": "string", "enum": ["price", "area", "subway"]},
                        "sort_order": {"type": "string", "enum": ["asc", "desc"]},
                        "page_size": {"type": "integer", "description": "每页条数，默认10，建议50"},
                    },
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_house_by_id",
                "description": "查房源详情，含tags、hidden_noise_level(安静/吵闹/中等)、utilities_type、orientation、elevator、community。多轮追加条件时用此逐个检查，禁止重新搜索。注意tag语义：养金毛→仅限小型犬=不符合应pass。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "house_id": {"type": "string", "description": "房源ID，如HF_123"},
                    },
                    "required": ["house_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_houses_by_community",
                "description": "按小区名查该小区可租房源，如「建清园(南区)」。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "community": {"type": "string", "description": "小区名，含括号，如建清园(南区)"},
                    },
                    "required": ["community"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "get_nearby_landmarks",
                "description": "查某小区周边地标。用于回答「附近有公园吗」「附近有商超吗」等。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "community": {"type": "string", "description": "小区名"},
                        "type": {"type": "string", "description": "地标类型：公园/商超/菜市场/医院/学校/餐饮/健身房"},
                    },
                    "required": ["community"],
                },
            },
        },
    ]

    action_tools = [
        {
            "type": "function",
            "function": {
                "name": "get_house_listings",
                "description": "查房源在链家/安居客/58同城各平台挂牌价，用于比价。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "house_id": {"type": "string"},
                    },
                    "required": ["house_id"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "rent_house",
                "description": "办理租房。必须指定house_id和listing_platform。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "house_id": {"type": "string"},
                        "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]},
                    },
                    "required": ["house_id", "listing_platform"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "terminate_rental",
                "description": "退租。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "house_id": {"type": "string"},
                        "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]},
                    },
                    "required": ["house_id", "listing_platform"],
                },
            },
        },
        {
            "type": "function",
            "function": {
                "name": "take_offline",
                "description": "下架房源。",
                "parameters": {
                    "type": "object",
                    "properties": {
                        "house_id": {"type": "string"},
                        "listing_platform": {"type": "string", "enum": ["链家", "安居客", "58同城"]},
                    },
                    "required": ["house_id", "listing_platform"],
                },
            },
        },
    ]

    if round_num >= 2:
        return search_tools + action_tools
    return search_tools
