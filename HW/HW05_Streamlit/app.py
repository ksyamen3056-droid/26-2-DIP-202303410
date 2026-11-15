import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO

st.set_page_config(
    page_title="Object Counting",
    page_icon="🔎",
    layout="wide"
)

st.title("Object Counting")
st.write("이미지를 업로드하면 Watershed 기반 객체 분리 및 계수를 수행합니다.")

def count_objects(image_bytes):
    # Uploaded image -> OpenCV BGR
    pil_image = Image.open(BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(pil_image)
    img_color = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    gray = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)

    # 1. Noise reduction (HW05)
    f_denoised = cv2.GaussianBlur(gray, (15, 15), 0)

    # 2. Otsu thresholding + morphological opening (HW05)
    th_val, f_mask = cv2.threshold(
        f_denoised, 0, 255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    kernel = np.ones((3, 3), np.uint8)
    f_mask = cv2.morphologyEx(
        f_mask, cv2.MORPH_OPEN, kernel, iterations=2
    )

    # 3. Distance transform and watershed markers (HW05)
    dist = cv2.distanceTransform(f_mask, cv2.DIST_L2, 5)
    if dist.max() == 0:
        return rgb, gray, f_denoised, f_mask, [], None, th_val

    _, sure_fg = cv2.threshold(dist, 0.35 * dist.max(), 255, 0)
    sure_fg = np.uint8(sure_fg)
    sure_bg = cv2.dilate(f_mask, kernel, iterations=3)
    unknown = cv2.subtract(sure_bg, sure_fg)

    nums, markers = cv2.connectedComponents(sure_fg)
    markers = markers + 1
    markers[unknown == 255] = 0

    labels_ws = cv2.watershed(img_color.copy(), markers)
    labels = labels_ws.copy()
    labels[labels <= 1] = 0
    labels[labels > 1] -= 1

    # 4. Keep coin-like components using the final HW05 criteria
    valid_objects = []
    for label_id in np.unique(labels):
        if label_id <= 0:
            continue

        mask = (labels == label_id).astype(np.uint8)
        area = int(cv2.countNonZero(mask))

        # HW05 minimum area criterion
        if area < 2000:
            continue

        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        if not contours:
            continue

        contour = max(contours, key=cv2.contourArea)
        perimeter = cv2.arcLength(contour, True)
        if perimeter == 0:
            continue

        circularity = (
            4 * np.pi * cv2.contourArea(contour) / (perimeter ** 2)
        )
        x, y, w, h = cv2.boundingRect(contour)
        aspect_ratio = w / h

        if circularity > 0.65 and 0.80 < aspect_ratio < 1.25:
            valid_objects.append({
                "label": int(label_id),
                "area": area,
                "mask": mask,
                "contour": contour
            })

    # Draw detected boundaries and indices
    result = img_color.copy()
    for number, obj in enumerate(valid_objects, start=1):
        cv2.drawContours(result, [obj["contour"]], -1, (0, 255, 0), 3)
        x, y, w, h = cv2.boundingRect(obj["contour"])
        cv2.putText(
            result, str(number), (x, max(y - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2,
            cv2.LINE_AA
        )

    result_rgb = cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
    return rgb, gray, f_denoised, f_mask, valid_objects, result_rgb, th_val


uploaded_file = st.file_uploader(
    "이미지 파일을 업로드하세요",
    type=["jpg", "jpeg", "png", "bmp", "tif", "tiff"]
)

if uploaded_file is not None:
    try:
        original, gray, denoised, binary_mask, objects, result, threshold = (
            count_objects(uploaded_file.getvalue())
        )

        st.subheader("분석 결과")
        col1, col2 = st.columns(2)
        with col1:
            st.image(original, caption="Input Image", use_container_width=True)
        with col2:
            if result is not None:
                st.image(result, caption="Object Counting Result", use_container_width=True)
            else:
                st.warning("객체를 검출할 수 없습니다. 다른 이미지를 시도해 주세요.")

        st.metric("검출된 객체 수", len(objects))

        if objects:
            st.write("객체별 픽셀 수")
            st.dataframe(
                [{"Object": i, "Pixels": obj["area"]}
                 for i, obj in enumerate(objects, start=1)],
                use_container_width=True,
                hide_index=True
            )

            largest = max(objects, key=lambda item: item["area"])
            smallest = min(objects, key=lambda item: item["area"])
            c1, c2 = st.columns(2)
            with c1:
                st.image(largest["mask"] * 255, caption=f"Largest object ({largest['area']} pixels)", clamp=True)
            with c2:
                st.image(smallest["mask"] * 255, caption=f"Smallest object ({smallest['area']} pixels)", clamp=True)

            # Download annotated result
            output = Image.fromarray(result)
            buffer = BytesIO()
            output.save(buffer, format="PNG")
            st.download_button(
                "결과 이미지 다운로드",
                data=buffer.getvalue(),
                file_name="object_counting_result.png",
                mime="image/png"
            )

        with st.expander("처리 과정 보기"):
            st.image(gray, caption="1. Grayscale Image", clamp=True)
            st.image(denoised, caption="2. Noise Reduction", clamp=True)
            st.image(binary_mask, caption=f"3. Binary Mask (Otsu threshold: {threshold:.0f})", clamp=True)

        st.caption(
            "처리 순서: Grayscale → Gaussian Blur → Otsu Thresholding "
            "→ Morphological Opening → Distance Transform → Watershed "
            "→ Connected Component 및 원형도/종횡비 필터링"
        )
    except Exception as e:
        st.error(f"이미지를 처리하는 중 오류가 발생했습니다: {e}")
else:
    st.info("위 입력란에서 이미지 파일을 선택하면 객체 계수를 시작합니다.")
