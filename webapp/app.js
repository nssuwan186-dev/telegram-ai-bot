const tg = window.Telegram?.WebApp;

if (!tg) {
  console.error("Telegram WebApp SDK not available");
}

const panels = {
  dashboard: document.getElementById("panel-dashboard"),
  slip: document.getElementById("panel-slip"),
  expense: document.getElementById("panel-expense"),
};

const statusBar = document.getElementById("statusBar");
const statusText = document.getElementById("statusText");

const setStatus = (message, type = "info") => {
  if (!statusBar || !statusText) return;
  statusBar.hidden = false;
  statusBar.classList.remove("status--error", "status--success");
  if (type === "error") {
    statusBar.classList.add("status--error");
  } else if (type === "success") {
    statusBar.classList.add("status--success");
  }
  statusText.textContent = message;
  if (tg?.HapticFeedback && type !== "info") {
    tg.HapticFeedback.impactOccurred(type === "success" ? "light" : "medium");
  }
};

const clearStatus = () => {
  if (!statusBar) return;
  statusBar.hidden = true;
  statusText.textContent = "";
  statusBar.classList.remove("status--error", "status--success");
};

const showPanel = (name) => {
  Object.entries(panels).forEach(([panelName, element]) => {
    if (!element) return;
    element.classList.toggle("is-visible", panelName === name);
  });
  clearStatus();
};

document
  .querySelectorAll("[data-open-section]")
  .forEach((control) =>
    control.addEventListener("click", () => {
      const target = control.dataset.openSection;
      if (target) {
        showPanel(target);
      }
    }),
  );

// Initialise Telegram Web App appearance
if (tg) {
  tg.expand();
  tg.ready();
  tg.MainButton.hide();
  tg.BackButton.hide();
}

// Slip verification form
const slipForm = document.getElementById("slipForm");

const withLoading = (button, fn) => async (...args) => {
  if (!button) return fn(...args);
  const original = button.innerHTML;
  button.disabled = true;
  button.innerHTML = "กำลังส่ง...";
  try {
    return await fn(...args);
  } finally {
    button.disabled = false;
    button.innerHTML = original;
  }
};

if (slipForm) {
  const submitButton = slipForm.querySelector("button[type='submit']");
  slipForm.addEventListener(
    "submit",
    withLoading(submitButton, async (event) => {
      event.preventDefault();
      if (!tg) {
        setStatus("ไม่สามารถเชื่อมต่อกับ Telegram ได้", "error");
        return;
      }

      const bookingId = slipForm.bookingId.value.trim();
      const reference = slipForm.reference.value.trim();
      const file = slipForm.slip.files[0];

      if (!bookingId) {
        setStatus("กรุณากรอกหมายเลขการจอง", "error");
        return;
      }

      if (!file) {
        setStatus("กรุณาอัปโหลดรูปสลิป", "error");
        return;
      }

      try {
        setStatus("กำลังอัปโหลดสลิป กรุณารอสักครู่...");
        const uploadResult = await tg.uploadFile(file);
        if (!uploadResult || !uploadResult.file_id) {
          throw new Error("ไม่สามารถอัปโหลดไฟล์ได้");
        }

        const payload = {
          type: "payment_slip",
          booking_id: bookingId,
          reference,
          file_id: uploadResult.file_id,
        };

        tg.sendData(JSON.stringify(payload));
        setStatus("ส่งข้อมูลให้บอทแล้ว กรุณารอผลการตรวจสอบภายในแชท", "success");
        slipForm.reset();
      } catch (error) {
        console.error("Slip upload failed", error);
        setStatus(error.message || "เกิดข้อผิดพลาดในการอัปโหลดสลิป", "error");
      }
    }),
  );
}

// Expense form
const expenseForm = document.getElementById("expenseForm");

if (expenseForm) {
  expenseForm.addEventListener("submit", (event) => {
    event.preventDefault();
    if (!tg) {
      setStatus("ไม่สามารถเชื่อมต่อกับ Telegram ได้", "error");
      return;
    }

    const amount = parseFloat(expenseForm.amount.value);
    const category = expenseForm.category.value;
    const note = expenseForm.note.value.trim();

    if (Number.isNaN(amount) || amount <= 0) {
      setStatus("กรุณากรอกจำนวนเงินที่ถูกต้อง", "error");
      return;
    }

    const payload = {
      type: "expense_entry",
      amount,
      category,
      note,
    };

    tg.sendData(JSON.stringify(payload));
    setStatus("บันทึกข้อมูลสำเร็จ ส่งให้บอทเรียบร้อย", "success");
    expenseForm.reset();
  });
}

showPanel("dashboard");

