import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
from folium.plugins import HeatMap, AntPath
import numpy as np
import requests
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
import time
from datetime import datetime
import math

# --- 1. Config ---
st.set_page_config(page_title="RoadRisk AI Center", page_icon="🧭", layout="wide")

# --- 🎨 UI & CSS Styling (ชุดแต่งหล่อ) ---
st.markdown("""
<style>
    /* 1. นำเข้าฟอนต์ Google (Prompt) ให้ดูทันสมัย */
    @import url('https://fonts.googleapis.com/css2?family=Prompt:wght@300;400;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Prompt', sans-serif;
    }

    /* 2. แต่งปุ่มกด (Modern Button) */
    div.stButton > button {
        background: linear-gradient(90deg, #FF416C 0%, #FF4B2B 100%);
        color: white;
        border: none;
        padding: 12px 28px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 30px; /* ขอบมน */
        box-shadow: 0 4px 15px rgba(255, 65, 108, 0.4); /* เงาฟุ้งๆ */
        transition: all 0.3s ease-in-out;
        font-weight: 600;
    }
    
    /* Effect ตอนเอาเมาส์ชี้ปุ่ม */
    div.stButton > button:hover {
        transform: translateY(-3px); /* ลอยขึ้นนิดนึง */
        box-shadow: 0 8px 25px rgba(255, 65, 108, 0.6);
        background: linear-gradient(90deg, #FF4B2B 0%, #FF416C 100%);
    }

    /* 3. แต่ง Logo (Text Gradient) */
    .logo-text {
        font-size: 50px;
        font-weight: 800;
        background: -webkit-linear-gradient(45deg, #FF416C, #FF4B2B);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
        margin-bottom: -10px;
    }
    
    .subtitle {
        font-size: 20px;
        color: #555;
        font-weight: 300;
        letter-spacing: 1px;
    }

    /* 4. แต่ง Card/Metric ให้ดูดีขึ้น */
    div[data-testid="stMetricValue"] {
        font-size: 28px;
        color: #FF4B2B;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# พิกัดและข้อมูลถนน
tambon_coords = {
    'ตลาดใหญ่': [7.8837, 98.3908], 'ตลาดเหนือ': [7.8872, 98.3860],
    'วิชิต': [7.8688, 98.3644],    'ฉลอง': [7.8344, 98.3375],
    'รัษฎา': [7.9045, 98.4026],    'ราไวย์': [7.7818, 98.3129],
    'กะรน': [7.8354, 98.2954],     'เกาะแก้ว': [7.9472, 98.3753],
    'ป่าตอง': [7.8960, 98.2955],   'กะทู้': [7.9224, 98.3360],
    'กมลา': [7.9547, 98.2858],     'เทพกระษัตรี': [8.0333, 98.3333],
    'ศรีสุนทร': [7.9750, 98.3500], 'เชิงทะเล': [7.9950, 98.3050],
    'ป่าคลอก': [8.0167, 98.4000],  'ไม้ขาว': [8.1333, 98.3000],
    'สาคู': [8.0833, 98.3000]
}

# --- 2. AI & Data Engine ---
@st.cache_resource
def train_ai_model(df_stats):
    training_data = []
    base_risks = df_stats.set_index('ตำบล')['ผู้ประสบภัย'].to_dict()
    for _ in range(5000):
        tambon = np.random.choice(list(base_risks.keys()))
        base_score = base_risks.get(tambon, 0)
        hour = np.random.randint(0, 24)
        is_rain = np.random.choice([0, 1], p=[0.8, 0.2])
        vehicle = np.random.choice(['Motorcycle', 'Car'], p=[0.8, 0.2])
        
        risk = (base_score / 1500 * 50) + (20 if is_rain else 0) + (15 if hour >= 18 or hour <= 5 else 0) + (10 if vehicle == 'Motorcycle' else 0)
        risk += np.random.normal(0, 2)
        training_data.append([tambon, hour, is_rain, vehicle, np.clip(risk, 0, 100)])
    
    df_train = pd.DataFrame(training_data, columns=['tambon', 'hour', 'is_rain', 'vehicle', 'risk_score'])
    le_tambon = LabelEncoder()
    le_vehicle = LabelEncoder()
    df_train['tambon_code'] = le_tambon.fit_transform(df_train['tambon'])
    df_train['vehicle_code'] = le_vehicle.fit_transform(df_train['vehicle'])
    
    X = df_train[['tambon_code', 'hour', 'is_rain', 'vehicle_code']]
    y = df_train['risk_score']
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    return model, le_tambon, le_vehicle

# --- 3. Routing & Helper Functions ---
@st.cache_data(ttl=3600)
def get_route_osrm(start_coords, end_coords):
    url = f"http://router.project-osrm.org/route/v1/driving/{start_coords[1]},{start_coords[0]};{end_coords[1]},{end_coords[0]}?overview=full&geometries=geojson"
    headers = {'User-Agent': 'Mozilla/5.0'}
    try:
        response = requests.get(url, headers=headers, timeout=5)
        if response.status_code == 200:
            data = response.json()
            if data['code'] == 'Ok':
                route = data['routes'][0]
                path_coords = [[p[1], p[0]] for p in route['geometry']['coordinates']]
                return {'distance_km': route['distance'] / 1000, 'duration_min': route['duration'] / 60, 'path': path_coords, 'status': 'ok'}
    except: pass
    
    # Fallback (เส้นตรง)
    R = 6371
    dLat = math.radians(end_coords[0] - start_coords[0])
    dLon = math.radians(end_coords[1] - start_coords[1])
    a = math.sin(dLat/2)**2 + math.cos(math.radians(start_coords[0])) * math.cos(math.radians(end_coords[0])) * math.sin(dLon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    dist_km = R * c
    duration_min = (dist_km / 30) * 60
    return {'distance_km': dist_km, 'duration_min': duration_min, 'path': [start_coords, end_coords], 'status': 'fallback'}

def find_nearest_tambon(lat, lng):
    min_dist = 9999
    nearest = None
    for name, coords in tambon_coords.items():
        dist = np.sqrt((coords[0]-lat)**2 + (coords[1]-lng)**2)
        if dist < min_dist: min_dist = dist; nearest = name
    return nearest

@st.cache_data(ttl=600)
def get_live_weather():
    try:
        url = "https://api.open-meteo.com/v1/forecast?latitude=7.8804&longitude=98.3922&current_weather=true"
        r = requests.get(url).json()
        code = r['current_weather']['weathercode']
        return {'is_rain': 1 if code >= 50 else 0, 'temp': r['current_weather']['temperature']}
    except: return {'is_rain': 0, 'temp': 30}

@st.cache_data
def load_data():
    try:
        df = pd.read_csv('phuket_powerbi_data.csv')
        df['lat'] = df['ตำบล'].map(lambda x: tambon_coords.get(x, [None, None])[0])
        df['lng'] = df['ตำบล'].map(lambda x: tambon_coords.get(x, [None, None])[1])
        return df.dropna(subset=['lat'])
    except: return None

def generate_advice_card(risk_score, tambon_name, is_rain, vehicle_type, hour):
    tips = []
    if risk_score > 70: tips.append("🚨 **อันตรายสูง:** หลีกเลี่ยงเส้นทางนี้หรือจอดพัก")
    elif risk_score > 40: tips.append("⚠️ **ระวังพิเศษ:** ลดความเร็วลงจากปกติ")
    else: tips.append("✅ **เดินทางปกติ:** ไม่ประมาท")
    
    if tambon_name in ['ป่าตอง', 'กะรน', 'กมลา', 'ราไวย์', 'สาคู']:
        tips.append("⛰️ **ทางเขา/โค้ง:** ใช้เกียร์ต่ำ ห้ามเลียเบรก")
    elif tambon_name in ['เทพกระษัตรี', 'ไม้ขาว', 'ศรีสุนทร']:
        tips.append("🛣️ **ทางหลวง:** ระวังรถกลับรถตัดหน้า")
    
    if is_rain: tips.append("🌧️ **ฝนตก:** ถนนลื่น! ลดความเร็ว 30% เว้นระยะเบรก")
    if vehicle_type == "Motorcycle": tips.append("🛵 **มอเตอร์ไซค์:** สวมหมวกกันน็อก ชิดซ้าย")
    if hour >= 18 or hour <= 5: tips.append("🌙 **กลางคืน:** ทัศนวิสัยต่ำ ระวังจุดมืด")
    return tips

# --- 4. Main App ---
df_tambon = load_data()

if df_tambon is not None:
    model, le_tambon, le_vehicle = train_ai_model(df_tambon)
    weather_now = get_live_weather()
    current_time = datetime.now()

    # Sidebar
    st.sidebar.title("📡 Control Center")
    if st.sidebar.button("🔄 สแกนพื้นที่เสี่ยง"): st.sidebar.success("Scanning complete.")

    # --- Header with Logo ---
    c_logo1, c_logo2 = st.columns([1, 4])
    with c_logo1:
        st.image("https://cdn-icons-png.flaticon.com/512/1048/1048313.png", width=120) 
    with c_logo2:
        st.markdown('<p class="logo-text">RoadRisk AI</p>', unsafe_allow_html=True)
        st.markdown('<p class="subtitle">เพื่อนร่วมทางอัจฉริยะ ภูเก็ต</p>', unsafe_allow_html=True)
    st.markdown("---")

    cols = st.columns(4)
    cols[0].metric("🌡️ อุณหภูมิ", f"{weather_now['temp']} °C")
    cols[1].metric("☁️ สภาพอากาศ", "ฝนตก" if weather_now['is_rain'] else "ปกติ")

    tab_map, tab_route, tab_line, tab_data = st.tabs(["🗺️ แผนที่โซนเสี่ยง", "🧭 วางแผนเส้นทาง (A-to-B)", "💬 LINE Simulator", "💾 ฐานข้อมูล"])

    # === TAB 1: Map ===
    with tab_map:
        m = folium.Map([7.9519, 98.3381], zoom_start=11)
        for _, row in df_tambon.iterrows():
            score = row['ผู้ประสบภัย'] + (200 if weather_now['is_rain'] else 0)
            color = '#FF0000' if score > 1000 else '#FF8C00' if score > 500 else '#32CD32'
            
            popup_txt = f"""
            <div style='font-family:sans-serif; width:200px'>
                <b>ต.{row['ตำบล']}</b><br>
                ความเสี่ยง: {score:.0f}<br>
                <hr>
                {'⛰️ ทางเขา/โค้ง' if row['ตำบล'] in ['ป่าตอง','กะรน'] else '🛣️ ทางปกติ'}
            </div>
            """
            folium.Circle([row['lat'], row['lng']], radius=row['ผู้ประสบภัย']*1.5, color=color, fill=True, fill_opacity=0.5, popup=folium.Popup(popup_txt)).add_to(m)
        st_folium(m, height=500)

    # === TAB 2: Route Planner ===
    with tab_route:
        st.subheader("📍 กำหนดเส้นทาง & รับคำแนะนำ")
        input_method = st.radio("เลือกวิธีระบุพิกัด:", ["📝 เลือกจากรายชื่อ", "👆 จิ้มบนแผนที่"], horizontal=True)
        
        start_coord, end_coord = None, None
        start_name, end_name = "", ""

        if "รายชื่อ" in input_method:
            c1, c2 = st.columns(2)
            with c1: start_name = st.selectbox("จุดเริ่มต้น (A)", df_tambon['ตำบล'].unique(), index=4)
            with c2: end_name = st.selectbox("ปลายทาง (B)", df_tambon['ตำบล'].unique(), index=8)
            if start_name and end_name: start_coord, end_coord = tambon_coords[start_name], tambon_coords[end_name]
        else:
            st.info("คลิกบนแผนที่ 2 ครั้ง (เริ่ม -> จบ)")
            if 'route_clicks' not in st.session_state: st.session_state.route_clicks = []
            if st.button("ล้างจุด"): st.session_state.route_clicks = []
            m_click = folium.Map([7.9519, 98.3381], zoom_start=11)
            for i, pt in enumerate(st.session_state.route_clicks): folium.Marker(pt, icon=folium.Icon(color='green' if i==0 else 'red')).add_to(m_click)
            out = st_folium(m_click, height=300, key="click_map")
            if out['last_clicked']:
                pt = [out['last_clicked']['lat'], out['last_clicked']['lng']]
                if not st.session_state.route_clicks or st.session_state.route_clicks[-1] != pt: st.session_state.route_clicks.append(pt)
            if len(st.session_state.route_clicks) >= 2:
                start_coord, end_coord = st.session_state.route_clicks[0], st.session_state.route_clicks[1]
                start_name, end_name = find_nearest_tambon(*start_coord), find_nearest_tambon(*end_coord)
                st.success(f"พิกัดใกล้: {start_name} -> {end_name}")

        st.divider()
        c_time, c_veh, c_btn = st.columns([1, 1, 1])
        with c_time: travel_time = st.time_input("เวลาเดินทาง", current_time)
        with c_veh: vehicle_type = st.radio("ยานพาหนะ", ["Motorcycle", "Car"])
        
        if c_btn.button("🚀 คำนวณความเสี่ยงและขอคำแนะนำ", type="primary"):
            if start_coord and end_coord:
                route_data = get_route_osrm(start_coord, end_coord)
                
                rain = weather_now['is_rain']
                v_code = le_vehicle.transform([vehicle_type])[0]
                t_end_code = le_tambon.transform([end_name])[0]
                risk_end = model.predict([[t_end_code, travel_time.hour, rain, v_code]])[0]
                advice_list = generate_advice_card(risk_end, end_name, rain, vehicle_type, travel_time.hour)

                st.session_state['calc_result'] = {
                    'route_data': route_data,
                    'trip_risk': risk_end,
                    'start_coord': start_coord, 'end_coord': end_coord,
                    'advice': advice_list
                }
            else:
                st.warning("กรุณาเลือกจุดเริ่มต้นและปลายทางให้ครบถ้วน")

        if 'calc_result' in st.session_state:
            res = st.session_state['calc_result']
            if res['route_data']['status'] == 'fallback':
                st.warning("⚠️ ไม่สามารถเชื่อมต่อระบบนำทางได้: แสดงระยะทางเส้นตรงและการประเมินเวลาคร่าวๆ แทน")

            m1, m2, m3 = st.columns(3)
            m1.metric("⏳ เวลาเดินทาง", f"{int(res['route_data']['duration_min'])} นาที")
            m2.metric("📏 ระยะทาง", f"{res['route_data']['distance_km']:.1f} กม.")
            m3.metric("⚠️ ความเสี่ยงปลายทาง", f"{res['trip_risk']:.1f}%", delta_color="inverse")
            
            st.markdown("### 💡 คำแนะนำการเดินทาง (Smart Advice)")
            with st.container():
                for tip in res['advice']: st.info(tip)

            m_res = folium.Map(location=[(res['start_coord'][0]+res['end_coord'][0])/2, (res['start_coord'][1]+res['end_coord'][1])/2], zoom_start=11)
            AntPath(res['route_data']['path'], color='blue', weight=5).add_to(m_res)
            folium.Marker(res['start_coord'], icon=folium.Icon(color='green', icon='play')).add_to(m_res)
            folium.Marker(res['end_coord'], icon=folium.Icon(color='red', icon='stop')).add_to(m_res)
            st_folium(m_res, height=400)

    # === TAB 3: LINE Chat ===
    with tab_line:
        st.subheader("💬 LINE Chatbot Demo")
        prompt = st.chat_input("พิมพ์ข้อความ (เช่น 'ไปป่าตอง')...")
        if prompt:
            st.chat_message("user").write(prompt)
            target_tambon = None
            for t in df_tambon['ตำบล'].unique():
                if t in prompt: target_tambon = t; break
            
            with st.chat_message("assistant"):
                if target_tambon:
                    rain = weather_now['is_rain']
                    v_code = le_vehicle.transform(['Motorcycle'])[0]
                    t_code = le_tambon.transform([target_tambon])[0]
                    risk = model.predict([[t_code, current_time.hour, rain, v_code]])[0]
                    advice = generate_advice_card(risk, target_tambon, rain, "Motorcycle", current_time.hour)
                    
                    st.markdown(f"**กำลังไป: {target_tambon}**")
                    st.markdown(f"⚠️ ความเสี่ยง: **{risk:.1f}%**")
                    st.markdown("**คำแนะนำ:**")
                    for tip in advice: st.markdown(f"- {tip}")
                else:
                    st.write("สวัสดีครับ! พิมพ์ชื่อตำบลที่อยากไปได้เลยครับ")

    with tab_data: st.dataframe(df_tambon)

else: st.error("❌ ไม่พบไฟล์ phuket_powerbi_data.csv")