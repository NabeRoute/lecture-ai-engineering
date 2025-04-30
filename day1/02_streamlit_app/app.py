# app.py
import streamlit as st
import ui                    # UIモジュール
import llm                   # LLMモジュール
import database              # データベースモジュール
import metrics               # 評価指標モジュール
import data                  # データモジュール
import torch
from transformers import pipeline
from config import MODEL_NAME
from huggingface_hub import HfFolder


# ---------------------------------------------------------------------------
# 1. テーマ切り替え用 CSS
# ---------------------------------------------------------------------------
_DARK_CSS = """
<style>
/* ベース背景 & 文字色 ----------------------------------------------------- */
.stApp, .stApp > header, .stApp > footer {
  background-color: #0E1117;
  color: #C9D1D9;
}
html, body, p, span, div, input, textarea, label {
  color: #C9D1D9 !important;
}

/* サイドバー -------------------------------------------------------------- */
section[data-testid="stSidebar"] {
  background-color: #161B22;
}

/* ボタン ------------------------------------------------------------------ */
button[kind="secondary"], button[kind="primary"] {
  background-color: #238636 !important;   /* GitHub green */
  color: #FFFFFF !important;
  border: none !important;
}
button[kind="secondary"]:hover, button[kind="primary"]:hover {
  background-color: #2EA043 !important;
}

/* エクスパンダ・ボックスの枠線 ------------------------------------------ */
div[class*="stExpander"] > summary {
  background-color: #161B22 !important;
  color: #C9D1D9 !important;
}
div[class*="stExpander"] {
  border: 1px solid #30363D !important;
}

/* metric コンポーネントの背景 -------------------------------------------- */
div[data-testid="stMetric"] {
  background-color: #161B22 !important;
  border: 1px solid #30363D !important;
  border-radius: 8px;
}
</style>
"""

# ライトモードに戻すときは、Over-Ride を外すだけで良いため空の <style> を流す
_LIGHT_CSS = "<style></style>"


def _apply_theme():
    """セッション状態の dark_mode に合わせて CSS を注入する。"""
    if st.session_state.get("dark_mode", False):
        st.markdown(_DARK_CSS, unsafe_allow_html=True)
    else:
        st.markdown(_LIGHT_CSS, unsafe_allow_html=True)


def _toggle_theme():
    """サイドバーのチェックボックス変更時に呼ばれるコールバック。"""
    st.session_state.dark_mode = not st.session_state.get("dark_mode", False)


# ---------------------------------------------------------------------------
# 2. Streamlit ページ設定
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Gemma Chatbot", layout="wide")


# ---------------------------------------------------------------------------
# 3. サイドバー (テーマトグル → 他の UI より先に描画)
# ---------------------------------------------------------------------------
with st.sidebar:
    st.title("ナビゲーション")

    # 初期化
    if "dark_mode" not in st.session_state:
        st.session_state.dark_mode = False

    # 🌙 ダークモード トグル
    st.checkbox(
        "🌙 ダークモード",
        value=st.session_state.dark_mode,
        key="dark_mode_checkbox",
        on_change=_toggle_theme,
    )

# 選択されたテーマを適用
_apply_theme()

# ---------------------------------------------------------------------------
# 4. 初期化処理
# ---------------------------------------------------------------------------
metrics.initialize_nltk()      # NLTK データ
database.init_db()             # DB テーブル作成
data.ensure_initial_data()     # サンプル投入


@st.cache_resource
def load_model():
    """LLM モデルをロードしてキャッシュ。"""
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        st.info(f"Using device: {device}")
        pipe = pipeline(
            "text-generation",
            model=MODEL_NAME,
            model_kwargs={"torch_dtype": torch.bfloat16},
            device=device,
        )
        st.success(f"モデル '{MODEL_NAME}' の読み込みに成功しました。")
        return pipe
    except Exception as e:
        st.error(f"モデル '{MODEL_NAME}' の読み込みに失敗しました: {e}")
        st.error("GPUメモリ不足の可能性があります。不要なプロセスを終了するか、より小さいモデルの使用を検討してください。")
        return None


pipe = llm.load_model()

# ---------------------------------------------------------------------------
# 5. ページ選択
# ---------------------------------------------------------------------------
if "page" not in st.session_state:
    st.session_state.page = "チャット"

page = st.sidebar.radio(
    "ページ選択",
    ["チャット", "履歴閲覧", "サンプルデータ管理"],
    key="page_selector",
    index=["チャット", "履歴閲覧", "サンプルデータ管理"].index(st.session_state.page),
    on_change=lambda: setattr(st.session_state, "page", st.session_state.page_selector),
)

# ---------------------------------------------------------------------------
# 6. メインコンテンツ
# ---------------------------------------------------------------------------
st.title("🤖 Gemma 2 Chatbot with Feedback")
st.write("Gemma モデルを使用したチャットボットです。回答に対してフィードバックを行えます。")
st.markdown("---")

if st.session_state.page == "チャット":
    if pipe:
        ui.display_chat_page(pipe)
    else:
        st.error("チャット機能を利用できません。モデルの読み込みに失敗しました。")
elif st.session_state.page == "履歴閲覧":
    ui.display_history_page()
elif st.session_state.page == "サンプルデータ管理":
    ui.display_data_page()

# ---------------------------------------------------------------------------
# 7. フッター
# ---------------------------------------------------------------------------
st.sidebar.markdown("---")
st.sidebar.info("開発者: [Your Name]")