import pytest
import os
import httpx
import time
from playwright.sync_api import Page, expect

BASE_URL = os.environ.get("E2E_BASE_URL", "http://localhost:5173")
BACKEND_URL = os.environ.get("E2E_BACKEND_URL", "http://localhost:8000")
VERCEL_BYPASS_SECRET = os.environ.get("VERCEL_BYPASS_SECRET", "")

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    if VERCEL_BYPASS_SECRET:
        return {
            **browser_context_args,
            "extra_http_headers": {
                "x-vercel-protection-bypass": VERCEL_BYPASS_SECRET
            }
        }
    return browser_context_args

#----warm up backend

@pytest.fixture(scope="session", autouse=True)
def warmup_backend():
    print(f"\nWarming up backend at {BACKEND_URL}")
    for i in range(12):
        try:
            response = httpx.get(f"{BACKEND_URL}/health", timeout=10)
            if response.status_code == 200:
                print("Backend is ready")
                return
        except:
            pass
        print(f"Attempt {i+1}: Backend not ready, waiting 15s...")
        time.sleep(15)
    raise RuntimeError("Failed to warm up within 3 minutes.")

#----wait for frontend to be ready

@pytest.fixture(scope="session", autouse=True)
def wait_for_frontend():
    print(f"\nWaiting for frontend at {BASE_URL}")
    headers = {}
    if VERCEL_BYPASS_SECRET:
        headers["x-vercel-protection-bypass"] = VERCEL_BYPASS_SECRET
    for i in range(24):
        try:
            response = httpx.get(BASE_URL, timeout=10, headers=headers, follow_redirects=False)
            if response.status_code == 200:
                print("Frontend is ready")
                return
        except Exception:
            pass
        print(f"Attempt {i+1}: Frontend not ready, waiting 5s...")
        time.sleep(5)
    raise RuntimeError("Frontend failed to become ready within 2 minutes")

#----test page loads

def test_page_loads(page: Page):
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    expect(page).to_have_title("Titanic Survival Predictor")

#----test form fields are visible

def test_form_fields_visible(page: Page):
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    expect(page.locator("select[name='pclass']")).to_be_visible()
    expect(page.locator("select[name='sex']")).to_be_visible()
    expect(page.locator("input[name='age']")).to_be_visible()
    expect(page.locator("input[name='fare']")).to_be_visible()
    expect(page.locator("input[name='sibsp']")).to_be_visible()
    expect(page.locator("input[name='parch']")).to_be_visible()
    expect(page.locator("select[name='embarked']")).to_be_visible()

#----dropdown options

def test_pclass_dropdown_options(page: Page):
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    options = page.locator("select[name='pclass'] option").all_text_contents()
    assert options == ["1st Class", "2nd Class", "3rd Class"]

def test_sex_dropdown_options(page: Page):
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    options = page.locator("select[name='sex'] option").all_text_contents()
    assert options == ["Female", "Male"]

def test_embarked_dropdown_options(page: Page):
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    options = page.locator("select[name='embarked'] option").all_text_contents()
    assert options == ["Southampton", "Cherbourg", "Queenstown"]

#----Happy paths

def test_predict_survived_happy_path(page: Page):
    # Capture browser console messages
    messages = []
    page.on("console", lambda msg: messages.append(f"{msg.type}: {msg.text}"))
    page.on("pageerror", lambda err: messages.append(f"ERROR: {err}"))

    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.select_option("select[name='pclass']", "1")
    page.select_option("select[name='sex']", "female")
    page.fill("input[name='age']", "29")
    page.fill("input[name='fare']", "100")
    page.fill("input[name='sibsp']", "0")
    page.fill("input[name='parch']", "0")
    page.select_option("select[name='embarked']", "S")
    page.get_by_role("button", name="Predict Survival").click()
    page.wait_for_timeout(5000)
    print(f"\nConsole messages: {messages}")
    expect(page.get_by_text("Prediction Result", exact=True)).to_be_visible(timeout=10000)
    expect(page.get_by_text("SURVIVED", exact=True)).to_be_visible()

def test_predict_perished_happy_path(page: Page):
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.select_option("select[name='pclass']", "3")
    page.select_option("select[name='sex']", "male")
    page.fill("input[name='age']", "40")
    page.fill("input[name='fare']", "5")
    page.fill("input[name='sibsp']", "0")
    page.fill("input[name='parch']", "0")
    page.select_option("select[name='embarked']", "S")
    page.get_by_role("button", name="Predict Survival").click()
    expect(page.get_by_text("Prediction Result", exact=True)).to_be_visible(timeout=10000)
    expect(page.get_by_text("PERISHED", exact=True)).to_be_visible()

def test_loading_state_appears(page: Page):
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="Predict Survival").click()
    expect(page.get_by_text("CALCULATING FATE...", exact=True)).to_be_visible(timeout=3000)
    time.sleep(5)
    page.screenshot(path="screenshot.png")

def test_result_shows_probability(page: Page):
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="Predict Survival").click()
    expect(page.get_by_text("Survival Probability")).to_be_visible(timeout=10000)

#----Sad paths

def test_backend_down_shows_error(page: Page):
    page.route("**/predict", lambda route: route.abort())
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="Predict Survival").click()
    expect(page.get_by_text("Failed to connect to the prediction API")).to_be_visible(timeout=5000)

def test_invalid_data_shows_validation_error(page: Page):
    page.route("**/predict", lambda route: route.fulfill(status=422))
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.get_by_role("button", name="Predict Survival").click()
    expect(page.get_by_text("Invalid passenger data")).to_be_visible(timeout=5000)

def test_negative_fare_shows_validation_error(page: Page):
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.fill("input[name='fare']", "-50")
    page.get_by_role("button", name="Predict Survival").click()
    expect(page.get_by_text("Invalid passenger data")).to_be_visible(timeout=10000)

def test_out_of_range_age_shows_validation_error(page: Page):
    page.goto(BASE_URL)
    page.wait_for_load_state("networkidle")
    page.fill("input[name='age']", "200")
    page.get_by_role("button", name="Predict Survival").click()
    expect(page.get_by_text("Invalid passenger data")).to_be_visible(timeout=10000)