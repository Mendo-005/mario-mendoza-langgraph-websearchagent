const form = document.querySelector("#chat-form");
const input = document.querySelector("#question");
const submit = document.querySelector("#submit");
const messages = document.querySelector("#messages");

function addMessage(text, role) {
  const message = document.createElement("div");
  message.className = `message ${role}`;
  message.textContent = text;
  messages.append(message);
  message.scrollIntoView({ behavior: "smooth", block: "nearest" });
}

// Intro envía el mensaje; Mayús + Intro inserta un salto de línea
input.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    form.requestSubmit();
  }
});

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const question = input.value.trim();
  if (!question) return;

  addMessage(question, "user");
  input.value = "";
  submit.disabled = true;
  submit.querySelector("span").textContent = "Buscando...";

  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ message: question }),
    });
    const payload = await response.json();
    if (!response.ok) throw new Error(payload.error || "No se pudo completar la consulta.");
    addMessage(payload.answer, "assistant");
  } catch (error) {
    addMessage(error.message || "No se pudo conectar con el servidor.", "error");
  } finally {
    submit.disabled = false;
    submit.querySelector("span").textContent = "Enviar";
    input.focus();
  }
});