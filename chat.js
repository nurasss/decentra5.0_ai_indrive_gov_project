const STORAGE_KEY = "zanai_chats_v3";

const botStates = {
  assistant: {
    title: "AI-ассистент для юриста",
    hint: "Ищет похожие нормы, explainability и оставляет решение за экспертом.",
  },
  review: {
    title: "Проверка нормы перед публикацией",
    hint: "Сверка с базой и поиск коллизий до выхода акта.",
  },
  summary: {
    title: "Краткое резюме проблемной нормы",
    hint: "Быстрый бриф: суть проблемы, связи, следующий шаг.",
  },
};

let conversations = [];
let activeChatId = null;

const analyzeWindow = document.getElementById("analyzeWindow");
const compareWindow = document.getElementById("compareWindow");
const analyzeInput = document.getElementById("analyzeInput");
const analyzeSendButton = document.getElementById("analyzeSend");
const compareInputA = document.getElementById("compareInputA");
const compareInputB = document.getElementById("compareInputB");
const compareSendButton = document.getElementById("compareSend");
const chatHistoryList = document.getElementById("chatHistoryList");
const newChatBtn = document.getElementById("newChatBtn");
const modePills = document.querySelectorAll(".gpt-mode-pill");
const modeHint = document.getElementById("modeHint");
const topbarTitle = document.getElementById("topbarTitle");
const gptSidebar = document.getElementById("gptSidebar");
const gptShell = document.querySelector(".gpt-shell");
const sidebarToggle = document.getElementById("sidebarToggle");
const sidebarOverlay = document.getElementById("sidebarOverlay");
const panePills = document.querySelectorAll(".gpt-pane-pill");

let activePaneView = "split";

function normalizeLegacyAssistantText(text) {
  if (typeof text !== "string") {
    return "";
  }

  const trimmed = text.trim();
  if (!trimmed.startsWith("Коротко:")) {
    return trimmed;
  }

  const parts = [];
  const shortMatch = trimmed.match(/Коротко:\s*([\s\S]*?)(?=\s*Проверка:|\s*Что нашлось:|\s*Источник:|$)/i);
  const checkMatch = trimmed.match(/Проверка:\s*([\s\S]*?)(?=\s*Что нашлось:|\s*Источник:|$)/i);
  const foundMatch = trimmed.match(/Что нашлось:\s*([\s\S]*?)(?=\s*Источник:|$)/i);
  const sourceMatch = trimmed.match(/Источник:\s*([\s\S]*?)$/i);

  if (shortMatch?.[1]) {
    parts.push(`Суть:\n${shortMatch[1].trim()}`);
  }
  if (foundMatch?.[1]) {
    parts.push(`Что нашлось:\n${foundMatch[1].trim()}`);
  }
  if (sourceMatch?.[1]) {
    parts.push(`Источник:\n${sourceMatch[1].trim()}`);
  }
  if (checkMatch?.[1]) {
    parts.push(`Проверка:\n${checkMatch[1].trim()}`);
  }

  return parts.length ? parts.join("\n\n") : trimmed;
}

function uid() {
  if (typeof crypto !== "undefined" && crypto.randomUUID) {
    return crypto.randomUUID();
  }
  return `id-${Date.now()}-${Math.random().toString(36).slice(2, 11)}`;
}

function loadState() {
  try {
    const raw = localStorage.getItem(STORAGE_KEY);
    if (!raw) {
      return { chats: [], activeId: null };
    }
    const data = JSON.parse(raw);
    return {
      chats: Array.isArray(data.chats) ? data.chats : [],
      activeId: data.activeId || null,
    };
  } catch {
    return { chats: [], activeId: null };
  }
}

function saveState() {
  try {
    localStorage.setItem(
      STORAGE_KEY,
      JSON.stringify({
        chats: conversations,
        activeId: activeChatId,
      })
    );
  } catch {
    /* ignore quota */
  }
}

function createEmptyChat() {
  return {
    id: uid(),
    title: "Новый чат",
    mode: "assistant",
    updatedAt: Date.now(),
    panes: {
      analyze: [],
      compare: [],
    },
  };
}

function getActiveChat() {
  return conversations.find((c) => c.id === activeChatId) || null;
}

function refreshTitle(chat) {
  const merged = [...(chat.panes?.analyze || []), ...(chat.panes?.compare || [])]
    .filter((m) => m.role === "user" && m.kind === "text")
    .sort((a, b) => (b.createdAt || 0) - (a.createdAt || 0));

  const latest = merged[0];
  if (latest && latest.text) {
    const t = latest.text.trim().replace(/\s+/g, " ");
    const base = t.length <= 42 ? t : `${t.slice(0, 40)}…`;
    const prefix = latest.pane === "compare" ? "[CMP] " : "[ANL] ";
    chat.title = `${prefix}${base}`;
  }
}

function syncModeUI() {
  const chat = getActiveChat();
  const mode = chat?.mode || "assistant";
  modePills.forEach((btn) => {
    btn.classList.toggle("active", btn.dataset.tab === mode);
  });
  const state = botStates[mode];
  if (modeHint && state) {
    modeHint.textContent = state.hint;
  }
  if (topbarTitle && state) {
    topbarTitle.textContent = state.title;
  }
}

function setChatMode(mode) {
  const chat = getActiveChat();
  if (!chat || !botStates[mode]) {
    return;
  }
  chat.mode = mode;
  chat.updatedAt = Date.now();
  saveState();
  syncModeUI();
}

modePills.forEach((btn) => {
  btn.addEventListener("click", () => setChatMode(btn.dataset.tab));
});

function renderHistory() {
  if (!chatHistoryList) {
    return;
  }

  const sorted = [...conversations].sort((a, b) => b.updatedAt - a.updatedAt);
  chatHistoryList.innerHTML = "";

  sorted.forEach((chat) => {
    const row = document.createElement("div");
    row.className = "gpt-history-row";
    if (chat.id === activeChatId) {
      row.classList.add("is-active");
    }

    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "gpt-history-item";
    btn.dataset.id = chat.id;
    btn.title = chat.title;

    const icon = document.createElement("span");
    icon.className = "gpt-history-item-icon";
    icon.setAttribute("aria-hidden", "true");
    icon.textContent = "💬";

    const label = document.createElement("span");
    label.className = "gpt-history-item-label";
    label.textContent = chat.title || "Без названия";

    btn.appendChild(icon);
    btn.appendChild(label);

    const del = document.createElement("button");
    del.type = "button";
    del.className = "gpt-history-del";
    del.dataset.delId = chat.id;
    del.setAttribute("aria-label", "Удалить чат");
    del.textContent = "×";

    btn.addEventListener("click", () => selectChat(chat.id));
    del.addEventListener("click", (e) => {
      e.stopPropagation();
      deleteChat(chat.id);
    });

    row.appendChild(btn);
    row.appendChild(del);
    chatHistoryList.appendChild(row);
  });
}

function selectChat(id) {
  if (!conversations.some((c) => c.id === id)) {
    return;
  }
  activeChatId = id;
  saveState();
  renderHistory();
  renderAllPanes();
  syncModeUI();
  closeSidebarMobile();
}

function deleteChat(id) {
  conversations = conversations.filter((c) => c.id !== id);
  if (conversations.length === 0) {
    const fresh = createEmptyChat();
    conversations.push(fresh);
    activeChatId = fresh.id;
  } else if (activeChatId === id) {
    activeChatId = conversations[0].id;
  }
  saveState();
  renderHistory();
  renderAllPanes();
  syncModeUI();
}

function newChat() {
  const chat = createEmptyChat();
  conversations.unshift(chat);
  activeChatId = chat.id;
  saveState();
  renderHistory();
  renderAllPanes();
  syncModeUI();
  analyzeInput?.focus();
  closeSidebarMobile();
}

newChatBtn?.addEventListener("click", newChat);

function renderEmptyState(container, title, subtitle) {
  const wrap = document.createElement("div");
  wrap.className = "gpt-empty";
  const logo = document.createElement("div");
  logo.className = "gpt-empty-logo";
  logo.textContent = title;
  const p = document.createElement("p");
  p.className = "gpt-empty-text";
  p.textContent = subtitle;
  wrap.appendChild(logo);
  wrap.appendChild(p);
  container.appendChild(wrap);
}

function buildTextBubble(role, text) {
  const row = document.createElement("div");
  row.className = `gpt-msg gpt-msg--${role}`;

  if (role === "assistant") {
    const av = document.createElement("div");
    av.className = "gpt-avatar";
    av.setAttribute("aria-hidden", "true");
    av.textContent = "ZA";
    row.appendChild(av);
  }

  const bubble = document.createElement("div");
  bubble.className = "gpt-bubble";
  bubble.textContent = role === "assistant" ? normalizeLegacyAssistantText(text) : text;
  row.appendChild(bubble);

  return row;
}

function buildSectionRow(title) {
  const row = document.createElement("div");
  row.className = "gpt-msg gpt-msg--assistant";
  const av = document.createElement("div");
  av.className = "gpt-avatar";
  av.setAttribute("aria-hidden", "true");
  av.textContent = "ZA";
  const bubble = document.createElement("div");
  bubble.className = "gpt-bubble gpt-bubble--section";
  const strong = document.createElement("strong");
  strong.textContent = title;
  bubble.appendChild(strong);
  row.appendChild(av);
  row.appendChild(bubble);
  return row;
}

function buildCardRow(item) {
  const row = document.createElement("div");
  row.className = "gpt-msg gpt-msg--assistant";
  const av = document.createElement("div");
  av.className = "gpt-avatar";
  av.setAttribute("aria-hidden", "true");
  av.textContent = "ZA";

  const bubble = document.createElement("div");
  bubble.className = "gpt-bubble gpt-bubble--card";

  const judge = item.judge || {
    label: "not_run",
    confidence: 0,
    routing: "pass",
    requires_human_review: false,
    step_1_extract_A: "",
    step_2_extract_B: "",
    step_3_compare: "",
  };

  const title = document.createElement("strong");
  title.textContent = item.title || "Найденная норма";
  bubble.appendChild(title);

  const meta = document.createElement("p");
  meta.className = "gpt-card-meta";
  meta.innerHTML =
    `Distance: <b>${Number(item.distance).toFixed(3)}</b> · baseline: <b>${item.baseline_label}</b> (${Number(item.baseline_score).toFixed(2)}) · judge: <b>${judge.label}</b> · routing: <b>${judge.routing}</b>`;
  bubble.appendChild(meta);

  if (item.url) {
    const link = document.createElement("a");
    link.href = item.url;
    link.target = "_blank";
    link.rel = "noreferrer";
    link.className = "gpt-card-link";
    link.textContent = item.url;
    bubble.appendChild(link);
  }

  if (judge.step_3_compare) {
    const reasoning = document.createElement("details");
    reasoning.className = "gpt-details";

    const summary = document.createElement("summary");
    summary.textContent = "Explainability";
    reasoning.appendChild(summary);

    const list = document.createElement("div");
    list.className = "gpt-details-body";
    list.innerHTML = `
      <p><b>A:</b> ${judge.step_1_extract_A || "—"}</p>
      <p><b>B:</b> ${judge.step_2_extract_B || "—"}</p>
      <p><b>Compare:</b> ${judge.step_3_compare || "—"}</p>
      <p><b>Human review:</b> ${judge.requires_human_review ? "Да" : "Нет"}</p>
    `;
    reasoning.appendChild(list);
    bubble.appendChild(reasoning);
  }

  const fragment = document.createElement("p");
  fragment.className = "gpt-card-text";
  fragment.textContent = item.text;
  bubble.appendChild(fragment);

  row.appendChild(av);
  row.appendChild(bubble);
  return row;
}

function getPaneMessages(chat, pane) {
  if (!chat?.panes) {
    return [];
  }
  return Array.isArray(chat.panes[pane]) ? chat.panes[pane] : [];
}

function renderPaneMessages(pane, container) {
  if (!container) {
    return;
  }

  container.innerHTML = "";
  const chat = getActiveChat();
  const messages = getPaneMessages(chat, pane);

  if (!chat || !messages.length) {
    if (pane === "analyze") {
      renderEmptyState(
        container,
        "Analyze",
        "Введите норму для анализа по базе и explainability."
      );
    } else {
      renderEmptyState(
        container,
        "Compare",
        "Введите в формате: A: ... B: ... чтобы сравнить две нормы."
      );
    }
    return;
  }

  messages.forEach((msg) => {
    if (msg.kind === "text") {
      const role = msg.role === "user" ? "user" : "assistant";
      container.appendChild(buildTextBubble(role, msg.text));
    } else if (msg.kind === "section") {
      container.appendChild(buildSectionRow(msg.text));
    } else if (msg.kind === "card") {
      container.appendChild(buildCardRow(msg.item));
    }
  });

  container.scrollTop = container.scrollHeight;
}

function renderAllPanes() {
  renderAnalyzePane();
  renderComparePane();
}

function renderAnalyzePane() {
  renderPaneMessages("analyze", analyzeWindow);
}

function renderComparePane() {
  renderPaneMessages("compare", compareWindow);
}

function applyPaneView() {
  if (!gptShell) {
    return;
  }
  gptShell.classList.remove("pane-split", "pane-analyze", "pane-compare");
  gptShell.classList.add(`pane-${activePaneView}`);
  panePills.forEach((pill) => {
    pill.classList.toggle("active", pill.dataset.pane === activePaneView);
  });
}

function setPaneView(view) {
  if (!["split", "analyze", "compare"].includes(view)) {
    return;
  }
  activePaneView = view;
  applyPaneView();
}

function pushMessage(pane, msg) {
  const chat = getActiveChat();
  if (!chat || !chat.panes?.[pane]) {
    return;
  }
  chat.panes[pane].push({ id: uid(), pane, createdAt: Date.now(), ...msg });
  chat.updatedAt = Date.now();
  refreshTitle(chat);
  saveState();
  renderHistory();
  renderAllPanes();
}

function pushAnalyzeMessage(msg) {
  pushMessage("analyze", msg);
}

function pushCompareMessage(msg) {
  pushMessage("compare", msg);
}

function pushMessagesBulk(pane, msgs) {
  const chat = getActiveChat();
  if (!chat || !chat.panes?.[pane] || !msgs.length) {
    return;
  }
  msgs.forEach((m) => {
    chat.panes[pane].push({ id: uid(), pane, createdAt: Date.now(), ...m });
  });
  chat.updatedAt = Date.now();
  refreshTitle(chat);
  saveState();
  renderHistory();
  renderAllPanes();
}

function pushAnalyzeMessages(msgs) {
  pushMessagesBulk("analyze", msgs);
}

function pushCompareMessages(msgs) {
  pushMessagesBulk("compare", msgs);
}

function buildAssistantSummary(payload) {
  if (payload.executive_summary || payload.analysis_mode) {
    const lines = [];
    const modeMap = {
      norm_review: "Режим: проверка нормы",
      title_search: "Режим: анализ акта",
      legal_question: "Режим: юридический поиск",
    };

    if (payload.analysis_mode && modeMap[payload.analysis_mode]) {
      lines.push(modeMap[payload.analysis_mode]);
    }
    if (payload.norm_status?.title) {
      lines.push(`Статус:\n${payload.norm_status.title}${payload.norm_status.summary ? `\n${payload.norm_status.summary}` : ""}`);
    }
    if (payload.executive_summary) {
      lines.push(`Сводка:\n${payload.executive_summary}`);
    }
    if (payload.primary_document_title) {
      lines.push(`Ключевой акт:\n${payload.primary_document_title}`);
    }
    if (payload.answer) {
      lines.push(`Объяснение:\n${payload.answer}`);
    }
    if (payload.judge_summary) {
      lines.push(`Проверка:\n${payload.judge_summary}`);
    }

    return lines.join("\n\n");
  }

  const lines = [];
  const top = payload.sources && payload.sources.length ? payload.sources[0] : null;

  if (payload.answer) {
    lines.push(`Суть:\n${payload.answer}`);
  }

  if (top) {
    const sourceParts = [top.title || "Найденный источник"];
    const articleMatch = (top.text || "").match(/Статья\s+[0-9-]+(?:-[0-9]+)?/i);
    if (articleMatch) {
      sourceParts.push(articleMatch[0]);
    }

    lines.push(`Что нашлось:\n${sourceParts.join(", ")}`);

    if (top.url) {
      lines.push(`Источник:\n${top.url}`);
    }
  } else {
    lines.push("Что нашлось:\nСистема не нашла достаточно близких фрагментов в текущей базе.");
  }

  if (payload.judge_summary) {
    lines.push(`Проверка:\n${payload.judge_summary}`);
  }

  return lines.join("\n\n");
}

function buildFindingText(items, emptyText) {
  if (!Array.isArray(items) || !items.length) {
    return emptyText;
  }

  return items
    .slice(0, 4)
    .map((item, index) => {
      const parts = [`${index + 1}. ${item.title || "Без названия"}`];
      if (item.explanation) {
        parts.push(item.explanation);
      }
      if (item.url) {
        parts.push(item.url);
      }
      return parts.join("\n");
    })
    .join("\n\n");
}

function buildCompareMessages(payload) {
  const judge = payload.judge || {};
  const verdict = payload.contradiction ? "Противоречие обнаружено" : "Явного противоречия не обнаружено";

  return [
    {
      role: "assistant",
      kind: "text",
      text: `Режим: прямое сравнение норм\n\nВердикт:\n${verdict}\n\nСводка:\n${payload.summary}`,
    },
    {
      kind: "section",
      text: "Сравнение A И B",
    },
    {
      role: "assistant",
      kind: "text",
      text:
        `A:\n${payload.text_a}\n\nB:\n${payload.text_b}\n\nBaseline:\n${payload.baseline_label} (${Number(payload.baseline_score).toFixed(2)})\n\nJudge:\n${judge.label || "unknown"}, confidence ${Number(judge.confidence || 0).toFixed(2)}, routing ${judge.routing || "pass"}`,
    },
    {
      kind: "section",
      text: "Explainability",
    },
    {
      role: "assistant",
      kind: "text",
      text:
        `A:\n${judge.step_1_extract_A || "—"}\n\nB:\n${judge.step_2_extract_B || "—"}\n\nCompare:\n${judge.step_3_compare || "—"}\n\nТребует ручной проверки:\n${judge.requires_human_review ? "Да" : "Нет"}`,
    },
  ];
}

function buildAnalysisMessages(payload) {
  const messages = [
    {
      role: "assistant",
      kind: "text",
      text: buildAssistantSummary(payload),
    },
  ];

  if (payload.norm_status?.title) {
    messages.push({ kind: "section", text: "Статус Нормы" });
    messages.push({
      role: "assistant",
      kind: "text",
      text: `${payload.norm_status.title}\n\n${payload.norm_status.summary || "Статус сформирован по верхним найденным актам."}`,
    });
  }

  if (Array.isArray(payload.possible_conflicts)) {
    messages.push({ kind: "section", text: "Потенциальные конфликты" });
    messages.push({
      role: "assistant",
      kind: "text",
      text: buildFindingText(
        payload.possible_conflicts,
        "Явных конфликтов среди верхних найденных актов не обнаружено."
      ),
    });
  }

  if (Array.isArray(payload.possible_duplicates)) {
    messages.push({ kind: "section", text: "Дубли И Версии" });
    messages.push({
      role: "assistant",
      kind: "text",
      text: buildFindingText(
        payload.possible_duplicates,
        "Явных дублей или близких редакций среди верхних результатов не обнаружено."
      ),
    });
  }

  if (Array.isArray(payload.staleness_signals)) {
    messages.push({ kind: "section", text: "Устаревание" });
    messages.push({
      role: "assistant",
      kind: "text",
      text: buildFindingText(
        payload.staleness_signals,
        "Сигналов утраты силы или явного устаревания среди верхних результатов не найдено."
      ),
    });
  }

  if (Array.isArray(payload.related_documents) && payload.related_documents.length) {
    messages.push({ kind: "section", text: "Связанные Акты" });
    payload.related_documents.slice(0, 4).forEach((item) => {
      messages.push({
        role: "assistant",
        kind: "text",
        text: `${item.title}\n\n${item.explanation}${item.url ? `\n\n${item.url}` : ""}`,
      });
    });
  }

  if (Array.isArray(payload.sources) && payload.sources.length) {
    messages.push({ kind: "section", text: "Фрагменты Для Проверки" });
    payload.sources.slice(0, 3).forEach((item) => {
      messages.push({ kind: "card", item });
    });
  }

  return messages;
}

function appendLoadingState(pane) {
  const chat = getActiveChat();
  const container = pane === "analyze" ? analyzeWindow : compareWindow;
  if (!container || !chat) {
    return;
  }

  const row = document.createElement("div");
  row.className = "gpt-msg gpt-msg--assistant gpt-msg--loading";
  row.id = `gptLoadingRow-${pane}`;
  const av = document.createElement("div");
  av.className = "gpt-avatar";
  av.textContent = "ZA";
  const bubble = document.createElement("div");
  bubble.className = "gpt-bubble gpt-bubble--typing";
  bubble.innerHTML = '<span class="gpt-dot"></span><span class="gpt-dot"></span><span class="gpt-dot"></span>';
  row.appendChild(av);
  row.appendChild(bubble);
  container.appendChild(row);
  container.scrollTop = container.scrollHeight;
}

function removeLoadingState(pane) {
  document.getElementById(`gptLoadingRow-${pane}`)?.remove();
}

function appendAnalyzeLoadingState() {
  appendLoadingState("analyze");
}

function removeAnalyzeLoadingState() {
  removeLoadingState("analyze");
}

function appendCompareLoadingState() {
  appendLoadingState("compare");
}

function removeCompareLoadingState() {
  removeLoadingState("compare");
}

async function analyzeNorm(query) {
  const response = await fetch("http://127.0.0.1:8000/api/analyze", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text: query,
      top_k: 2,
    }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail;
    let message = "Backend request failed.";

    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail
        .map((item) => item?.msg || JSON.stringify(item))
        .join("; ");
    } else if (detail && typeof detail === "object") {
      message = detail.msg || JSON.stringify(detail);
    }

    throw new Error(message);
  }

  return response.json();
}

async function compareNorms(textA, textB) {
  const response = await fetch("http://127.0.0.1:8000/api/compare", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({
      text_a: textA,
      text_b: textB,
    }),
  });

  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const detail = payload.detail;
    let message = "Backend request failed.";

    if (typeof detail === "string") {
      message = detail;
    } else if (Array.isArray(detail)) {
      message = detail.map((item) => item?.msg || JSON.stringify(item)).join("; ");
    } else if (detail && typeof detail === "object") {
      message = detail.msg || JSON.stringify(detail);
    }

    throw new Error(message);
  }

  return response.json();
}

async function handleAnalyzeSend() {
  const value = analyzeInput.value.trim();
  if (!value) {
    return;
  }

  const chat = getActiveChat();
  if (!chat) {
    return;
  }

  pushAnalyzeMessage({ role: "user", kind: "text", text: value });
  analyzeInput.value = "";
  autoResizeTextarea(analyzeInput);
  analyzeSendButton.disabled = true;
  if (compareSendButton) {
    compareSendButton.disabled = true;
  }

  appendAnalyzeLoadingState();

  try {
    const payload = await analyzeNorm(value);
    removeAnalyzeLoadingState();
    pushAnalyzeMessages(buildAnalysisMessages(payload));
  } catch (error) {
    removeAnalyzeLoadingState();
    pushAnalyzeMessages([
      { role: "assistant", kind: "text", text: `Ошибка при обращении к API: ${error.message}` },
    ]);
  } finally {
    analyzeSendButton.disabled = false;
    if (compareSendButton) {
      compareSendButton.disabled = false;
    }
  }
}

async function handleCompareSend() {
  const textA = compareInputA.value.trim();
  const textB = compareInputB.value.trim();
  if (!textA || !textB) {
    return;
  }

  const chat = getActiveChat();
  if (!chat) {
    return;
  }

  pushCompareMessage({
    role: "user",
    kind: "text",
    text: `A: ${textA}\n\nB: ${textB}`,
  });
  compareInputA.value = "";
  compareInputB.value = "";
  autoResizeTextarea(compareInputA);
  autoResizeTextarea(compareInputB);
  compareSendButton.disabled = true;
  if (analyzeSendButton) {
    analyzeSendButton.disabled = true;
  }

  appendCompareLoadingState();

  try {
    const payload = await compareNorms(textA, textB);
    removeCompareLoadingState();
    pushCompareMessages(buildCompareMessages(payload));
  } catch (error) {
    removeCompareLoadingState();
    pushCompareMessages([
      { role: "assistant", kind: "text", text: `Ошибка при обращении к API: ${error.message}` },
    ]);
  } finally {
    compareSendButton.disabled = false;
    if (analyzeSendButton) {
      analyzeSendButton.disabled = false;
    }
  }
}

analyzeSendButton?.addEventListener("click", handleAnalyzeSend);
compareSendButton?.addEventListener("click", handleCompareSend);

function autoResizeTextarea(target) {
  if (!target) {
    return;
  }
  target.style.height = "auto";
  target.style.height = `${Math.min(target.scrollHeight, 200)}px`;
}

analyzeInput?.addEventListener("input", () => autoResizeTextarea(analyzeInput));
compareInputA?.addEventListener("input", () => autoResizeTextarea(compareInputA));
compareInputB?.addEventListener("input", () => autoResizeTextarea(compareInputB));
analyzeInput?.addEventListener("focus", () => setPaneView("analyze"));
analyzeInput?.addEventListener("input", () => setPaneView("analyze"));
compareInputA?.addEventListener("focus", () => setPaneView("compare"));
compareInputA?.addEventListener("input", () => setPaneView("compare"));
compareInputB?.addEventListener("focus", () => setPaneView("compare"));
compareInputB?.addEventListener("input", () => setPaneView("compare"));

panePills.forEach((pill) => {
  pill.addEventListener("click", () => setPaneView(pill.dataset.pane));
});

analyzeInput?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    handleAnalyzeSend();
  }
});

compareInputB?.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    handleCompareSend();
  }
});

function openSidebarMobile() {
  gptSidebar?.classList.add("is-open");
  sidebarOverlay?.classList.add("is-visible");
  sidebarToggle?.setAttribute("aria-expanded", "true");
}

function closeSidebarMobile() {
  gptSidebar?.classList.remove("is-open");
  sidebarOverlay?.classList.remove("is-visible");
  sidebarToggle?.setAttribute("aria-expanded", "false");
}

sidebarToggle?.addEventListener("click", () => {
  if (gptSidebar?.classList.contains("is-open")) {
    closeSidebarMobile();
  } else {
    openSidebarMobile();
  }
});

sidebarOverlay?.addEventListener("click", closeSidebarMobile);

document.addEventListener("keydown", (event) => {
  if (event.key === "Escape") {
    closeSidebarMobile();
  }
});

function init() {
  const state = loadState();
  if (!state.chats.length) {
    conversations = [createEmptyChat()];
    activeChatId = conversations[0].id;
  } else {
    conversations = state.chats.map((c) => ({
      ...c,
      mode: c.mode || "assistant",
      panes: {
        analyze: Array.isArray(c.panes?.analyze)
          ? c.panes.analyze.map((message) => {
              if (message?.role === "assistant" && message?.kind === "text") {
                return {
                  ...message,
                  pane: "analyze",
                  createdAt: message.createdAt || Date.now(),
                  text: normalizeLegacyAssistantText(message.text),
                };
              }
              return {
                ...message,
                pane: message.pane || "analyze",
                createdAt: message.createdAt || Date.now(),
              };
            })
          : Array.isArray(c.messages)
            ? c.messages.map((m) => ({
                ...m,
                pane: "analyze",
                createdAt: m.createdAt || Date.now(),
              }))
            : [],
        compare: Array.isArray(c.panes?.compare)
          ? c.panes.compare.map((m) => ({
              ...m,
              pane: "compare",
              createdAt: m.createdAt || Date.now(),
            }))
          : [],
      },
    }));
    activeChatId = state.activeId && state.chats.some((c) => c.id === state.activeId)
      ? state.activeId
      : conversations[0].id;
  }
  saveState();
  renderHistory();
  renderAllPanes();
  syncModeUI();
  applyPaneView();
}

init();
