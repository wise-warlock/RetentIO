import os
import google.generativeai as genai
from dotenv import load_dotenv

# 1. Load API Key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Cấu hình Gemini
if api_key:
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel('gemini-2.0-flash')
else:
    model = None

def decide_strategy(metrics):
    """
    Logic xác định chiến lược (Rule-based).
    Input: dict metrics từ inference
    Output: Action code, Message Header, Color
    """
    uplift = metrics.get('uplift_segment', 'Lost Causes')
    clv = metrics.get('clv_segment', 'Low Value')
    
    if uplift == 'Sleeping Dogs':
        return 'DO_NOT_DISTURB', '⛔ CẢNH BÁO: Nguy cơ cao - Không làm phiền', 'grey'
    
    if uplift == 'Persuadables':
        if clv == 'High Value':
            return 'RETENTION_VIP', '🎁 ƯU TIÊN SỐ 1: Giữ chân VIP (High Touch)', 'red'
        else:
            return 'RETENTION_MASS', '📧 Ưu tiên TB: Gửi Voucher tự động (Low Touch)', 'orange'
            
    if uplift == 'Sure Things':
        return 'CROSS_SELL', '🟢 Khách an toàn: Đề xuất Cross-sell', 'green'
        
    return 'IGNORE', '🔵 Hiệu quả thấp: Không ưu tiên', 'blue'

def generate_template_message(action, info):
    """
    Hàm dự phòng: Tạo nội dung bằng mẫu có sẵn (Khi AI hết quota)
    """
    name = str(info['name']).title()
    balance_str = f"{info['balance']:,.0f}"
    
    if action == 'RETENTION_VIP':
        return (
            f"Kính gửi Quý khách {name},\n\n"
            f"Vinbank trân trọng gửi tặng Quý khách đặc quyền 'Miễn phí thường niên trọn đời' "
            f"và set quà tặng tri ân cao cấp.\n\n"
            f"Chuyên viên tư vấn sẽ liên hệ Quý khách trong ít phút tới."
        )
    elif action == 'RETENTION_MASS':
        return (
            f"Chào {name} ơi! 👋\n\n"
            f"Vinbank tặng riêng bạn voucher hoàn tiền 50K cho hóa đơn điện/nước tháng này.\n\n"
            f"👉 Vào App dùng ngay kẻo lỡ nhé!"
        )
    elif action == 'CROSS_SELL':
        return (
            f"Gợi ý tài chính cho {name}:\n\n"
            f"Với số dư khả dụng {balance_str} VNĐ, bạn có thể tối ưu lợi nhuận bằng "
            f"Gói Tiết kiệm Online lãi suất 6.5%/năm.\n\n"
            f"📱 Mở sổ chỉ với 3 chạm trên App!"
        )
    return "(Hệ thống quyết định không gửi tin nhắn)"

def generate_message(action, info):
    """
    Hàm chính: Gọi AI trước -> Nếu lỗi thì gọi Template dự phòng
    """
    # 1. Kiểm tra Action trước, nếu là nhóm bỏ qua thì return luôn đỡ tốn quota
    if action in ['DO_NOT_DISTURB', 'IGNORE']:
        return "(AI Agent quyết định im lặng)"

    # 2. Thử gọi Gemini API
    if model:
        try:
            prompt = f"""
            Bạn là CSKH Vinbank. Viết tin nhắn ngắn (max 40 từ) cho khách: {info['name']}, nghề: {info['occupation']}.
            Mục tiêu: {action}.
            - RETENTION_VIP: Trang trọng, tặng quà VIP.
            - RETENTION_MASS: Thân thiện, tặng 50k.
            - CROSS_SELL: Bán thêm thẻ/tiết kiệm.
            Giọng văn tự nhiên, có emoji.
            """
            response = model.generate_content(prompt)
            return response.text # Thành công trả về luôn
            
        except Exception as e:
            # Bắt lỗi Quota hoặc lỗi mạng -> Chuyển sang phương án B
            print(f"⚠️ Gemini API Error (Fallback to Template): {str(e)}")
            return generate_template_message(action, info) + "\n\n(Nội dung được tạo bởi Template dự phòng do AI quá tải)"
    
    # 3. Nếu không có model (chưa config key) -> Dùng Template
    return generate_template_message(action, info)