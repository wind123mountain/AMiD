import os
import json
import glob
import argparse

def aggregate_all_results(directory_path, output_filepath):
    compiled_data = {}
    search_pattern = os.path.join(directory_path, "**", "*results*.json")
    file_paths = glob.glob(search_pattern, recursive=True)
    
    if not file_paths:
        print(f"⚠️ Không tìm thấy file JSON nào chứa chữ 'result' trong: {directory_path}")
        return

    print(f"🔍 Tìm thấy {len(file_paths)} file. Đang tiến hành trích xuất...")

    for file_path in file_paths:
        filename = os.path.basename(file_path)
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
                if "results" in data:
                    compiled_data[filename] = data["results"]
                else:
                    print(f"⏩ Bỏ qua {filename}: Không tìm thấy trường ['results']")
        except Exception as e:
            print(f"❌ Lỗi khi đọc {filename}: {e}")

    if compiled_data:
        try:
            with open(output_filepath, 'w', encoding='utf-8') as out_f:
                json.dump(compiled_data, out_f, indent=4, ensure_ascii=False)
            print(f"\n✅ Đã tổng hợp thành công {len(compiled_data)} files.")
            print(f"📁 File output được lưu tại: {os.path.abspath(output_filepath)}")
        except Exception as e:
            print(f"❌ Lỗi khi ghi file: {e}")
    else:
        print("\n⚠️ Không có dữ liệu hợp lệ.")

if __name__ == "__main__":
    # Khởi tạo parser để nhận tham số từ Terminal / Bash Script
    parser = argparse.ArgumentParser(description="Tổng hợp các file kết quả JSON.")
    
    # Định nghĩa tham số -i (input) và -o (output)
    parser.add_argument("-i", "--input", type=str, required=True, help="Đường dẫn thư mục chứa các file JSON")
    parser.add_argument("-o", "--output", type=str, default="summary_all_results.json", help="Đường dẫn file JSON đầu ra")
    
    args = parser.parse_args()
    
    aggregate_all_results(args.input, args.output)