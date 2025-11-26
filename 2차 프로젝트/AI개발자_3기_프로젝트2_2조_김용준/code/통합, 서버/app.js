// static/app.js

// --- DOM 요소 ---
const categorySelect = document.getElementById("categorySelect");
const keywordInput = document.getElementById("keywordInput");
const btnGetQuestions = document.getElementById("btnGetQuestions");
const keywordMsg = document.getElementById("keywordMsg");

const qaListDiv = document.getElementById("qaList");
const btnExampleIntent = document.getElementById("btnExampleIntent");
const questionMsg = document.getElementById("questionMsg");

const exampleAnswerBox = document.getElementById("exampleAnswerBox");
const questionIntentBox = document.getElementById("questionIntentBox");

const usernameInput = document.getElementById("usernameInput");

const btnStartRec1 = document.getElementById("btnStartRec1");
const btnStopRec1 = document.getElementById("btnStopRec1");
const recordStatus1 = document.getElementById("recordStatus1");
const audioPreview1 = document.getElementById("audioPreview1");
const audioFileInput1 = document.getElementById("audioFileInput1");
const btnSubmitFirst = document.getElementById("btnSubmitFirst");
const submitStatus1 = document.getElementById("submitStatus1");

const firstResultBox = document.getElementById("firstResultBox");
const summaryBox = document.getElementById("summaryBox");
const modelAnswerBox = document.getElementById("modelAnswerBox");

const btnStartRec2 = document.getElementById("btnStartRec2");
const btnStopRec2 = document.getElementById("btnStopRec2");
const recordStatus2 = document.getElementById("recordStatus2");
const audioPreview2 = document.getElementById("audioPreview2");
const audioFileInput2 = document.getElementById("audioFileInput2");
const btnSubmitSecond = document.getElementById("btnSubmitSecond");
const submitStatus2 = document.getElementById("submitStatus2");
const secondResultBox = document.getElementById("secondResultBox");

const leaderboardFirstBody = document.querySelector("#leaderboardFirst tbody");
const leaderboardSecondBody = document.querySelector("#leaderboardSecond tbody");

// --- 상태 변수 ---
let currentQuestions = [];
let selectedQuestion = null;
let selectedQuestionIndex = null;  // 1~5 번호

let exampleAnswer = "";
let questionIntent = "";

let mediaRecorder1 = null;
let audioChunks1 = [];
let recordedBlob1 = null;

let mediaRecorder2 = null;
let audioChunks2 = [];
let recordedBlob2 = null;

let currentAttemptId = null;
let latestFirstScore = null;
let latestSecondScore = null;

// ===================================================
// 1. 카테고리 번호 + 키워드 → 질문 5개
// ===================================================
btnGetQuestions.addEventListener("click", async () => {
    const keyword = keywordInput.value.trim();
    if (!keyword) {
        keywordMsg.textContent = "키워드를 입력하세요.";
        return;
    }

    const categoryIdStr = categorySelect.value;
    const categoryLabel = categorySelect.options[categorySelect.selectedIndex].text;
    const categoryId = parseInt(categoryIdStr, 10);

    keywordMsg.textContent = "질문을 불러오는 중입니다...";
    const formData = new FormData();
    formData.append("category_id", categoryId);
    formData.append("category_label", categoryLabel);
    formData.append("keyword", keyword);

    try {
        const res = await fetch("/api/generate_questions", {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error("Server error");
        const data = await res.json();
        currentQuestions = data.questions || [];
        renderQuestionList();
        keywordMsg.textContent = "질문이 로드되었습니다. 하나를 선택하세요.";
        selectedQuestion = null;
        selectedQuestionIndex = null;
        btnExampleIntent.disabled = true;
        exampleAnswerBox.textContent = "아직 예시 답변이 없습니다.";
        questionIntentBox.textContent = "아직 질문 의도가 없습니다.";
    } catch (err) {
        console.error(err);
        keywordMsg.textContent = "질문 로드 중 오류가 발생했습니다.";
    }
});

function renderQuestionList() {
    if (!currentQuestions.length) {
        qaListDiv.textContent = "질문이 없습니다.";
        return;
    }
    qaListDiv.innerHTML = "";
    currentQuestions.forEach((q, idx) => {
        const div = document.createElement("div");
        div.className = "qa-item";
        div.dataset.index = idx;

        const qEl = document.createElement("div");
        qEl.className = "qa-question";
        // 앞에 1~5 번호 붙여서 표시
        qEl.textContent = `${idx + 1}. ${q}`;

        div.appendChild(qEl);

        div.addEventListener("click", () => {
            selectQuestion(idx);
        });

        qaListDiv.appendChild(div);
    });
}

function selectQuestion(idx) {
    const items = qaListDiv.querySelectorAll(".qa-item");
    items.forEach((el) => el.classList.remove("selected"));

    const selected = qaListDiv.querySelector(`.qa-item[data-index="${idx}"]`);
    if (selected) selected.classList.add("selected");

    selectedQuestion = currentQuestions[idx];   // 질문 텍스트
    selectedQuestionIndex = idx + 1;           // 1~5 번호
    questionMsg.textContent = `선택된 질문 (${selectedQuestionIndex}): ${selectedQuestion}`;
    btnExampleIntent.disabled = false;
}

// ===================================================
// 2. 질문 번호(1~5) → 예시 답변 + 질문 의도
// ===================================================
btnExampleIntent.addEventListener("click", async () => {
    if (!selectedQuestion || !selectedQuestionIndex) {
        questionMsg.textContent = "먼저 질문을 선택하세요.";
        return;
    }

    questionMsg.textContent = "예시 답변과 질문 의도를 생성 중입니다...";
    const formData = new FormData();
    formData.append("question_index", selectedQuestionIndex);

    try {
        const res = await fetch("/api/generate_example_intent", {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error("Server error");
        const data = await res.json();
        exampleAnswer = data.example_answer || "";
        questionIntent = data.question_intent || "";

        exampleAnswerBox.textContent = exampleAnswer || "예시 답변이 비어 있습니다.";
        questionIntentBox.textContent = questionIntent || "질문 의도가 비어 있습니다.";
        questionMsg.textContent = "예시 답변과 질문 의도가 생성되었습니다. 1차 발화를 녹음하세요.";
        btnSubmitFirst.disabled = false;
    } catch (err) {
        console.error(err);
        questionMsg.textContent = "예시 답변/질문 의도 생성 중 오류가 발생했습니다.";
    }
});

// ===================================================
// 3. 첫 번째 발화 녹음/업로드
// ===================================================
btnStartRec1.addEventListener("click", async () => {
    if (!selectedQuestion) {
        recordStatus1.textContent = "먼저 질문을 선택하고 예시 답변을 확인하세요.";
        return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        recordStatus1.textContent = "브라우저에서 마이크를 지원하지 않습니다.";
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder1 = new MediaRecorder(stream);
        audioChunks1 = [];

        mediaRecorder1.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks1.push(event.data);
        };
        mediaRecorder1.onstop = () => {
            recordedBlob1 = new Blob(audioChunks1, { type: "audio/webm" });
            audioChunks1 = [];
            audioPreview1.src = URL.createObjectURL(recordedBlob1);
            audioPreview1.style.display = "block";
            recordStatus1.textContent = "첫 번째 발화 녹음이 완료되었습니다.";
        };

        mediaRecorder1.start();
        recordStatus1.textContent = "녹음 중입니다. 자신의 답변을 말하세요...";
        btnStartRec1.disabled = true;
        btnStopRec1.disabled = false;
        audioPreview1.style.display = "none";
        recordedBlob1 = null;
    } catch (err) {
        console.error(err);
        recordStatus1.textContent = "마이크 사용 중 오류가 발생했습니다.";
    }
});

btnStopRec1.addEventListener("click", () => {
    if (mediaRecorder1 && mediaRecorder1.state === "recording") {
        mediaRecorder1.stop();
        btnStartRec1.disabled = false;
        btnStopRec1.disabled = true;
    }
});

// ===================================================
// 4. 첫 번째 평가 요청
// ===================================================
btnSubmitFirst.addEventListener("click", async () => {
    const username = usernameInput.value.trim();
    const keyword = keywordInput.value.trim();
    const categoryLabel = categorySelect.options[categorySelect.selectedIndex].text;

    if (!username) {
        submitStatus1.textContent = "이름을 입력하세요.";
        return;
    }
    if (!selectedQuestion) {
        submitStatus1.textContent = "질문을 먼저 선택하세요.";
        return;
    }
    if (!exampleAnswer) {
        submitStatus1.textContent = "예시 답변을 먼저 생성하세요.";
        return;
    }

    // 오디오 소스 선택 (녹음 > 업로드)
    let audioFileToSend = null;
    if (recordedBlob1) {
        audioFileToSend = new File([recordedBlob1], "first.webm", { type: "audio/webm" });
    } else if (audioFileInput1.files.length > 0) {
        audioFileToSend = audioFileInput1.files[0];
    }

    if (!audioFileToSend) {
        submitStatus1.textContent = "첫 번째 발화를 녹음하거나 파일을 업로드하세요.";
        return;
    }

    submitStatus1.textContent = "첫 번째 발화를 평가 중입니다...";
    btnSubmitFirst.disabled = true;

    const formData = new FormData();
    formData.append("username", username);
    formData.append("category", categoryLabel);    // '1.IT(정보,통신)' 형식 전체
    formData.append("keyword", keyword);
    formData.append("question", selectedQuestion);
    formData.append("example_answer", exampleAnswer);
    formData.append("question_intent", questionIntent);
    formData.append("audio", audioFileToSend);

    try {
        const res = await fetch("/api/first_evaluate", {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error("Server error");
        const data = await res.json();

        // 1차 시도 ID / 점수 저장
        currentAttemptId = data.attempt_id;
        latestFirstScore = data.first_score;

        // 결과/리더보드 렌더링
        renderFirstResult(data);
        renderLeaderboardFirst(data.first_leaderboard, data.first_score, username);

        // ---- 발화 요약(강점/보완점/논리성 평가/수정 제안/점수) 표시 ----
        const strengths = data.strengths || "";
        const weaknesses = data.weaknesses || "";
        const logicEval = data.logic_eval || "";
        const suggestions = data.suggestions || "";
        const logicScoreText = data.logic_score_text || "";

        if (
            !strengths &&
            !weaknesses &&
            !logicEval &&
            !suggestions &&
            !logicScoreText
        ) {
            summaryBox.textContent = "요약이 비어 있습니다.";
        } else {
            const parts = [];
            if (strengths) parts.push("[강점]\n" + strengths);
            if (weaknesses) parts.push("[보완점]\n" + weaknesses);
            if (logicEval) parts.push("[논리성 평가]\n" + logicEval);
            if (suggestions) parts.push("[수정 제안]\n" + suggestions);
            if (logicScoreText) parts.push("[점수]\n" + logicScoreText);
            summaryBox.textContent = parts.join("\n\n");
        }

        // 모범 답변 박스 (LLM이 만든 best_answer)
        modelAnswerBox.textContent =
            data.model_answer || data.best_answer || "모범 답변이 비어 있습니다.";

        submitStatus1.textContent = "첫 번째 평가가 완료되었습니다.";

        // 2차 발화 UI 활성화
        btnStartRec2.disabled = false;
        btnStopRec2.disabled = true;
        audioFileInput2.disabled = false;
        btnSubmitSecond.disabled = false;
    } catch (err) {
        console.error(err);
        submitStatus1.textContent = "첫 번째 평가 중 오류가 발생했습니다.";
    } finally {
        btnSubmitFirst.disabled = false;
    }
});

function renderFirstResult(data) {
    const {
        first_score,
        first_rank,
        first_total,
        first_percentile,
        logic_score,
        stutter_count,
        filler_count,
    } = data;

    if (
        first_score === undefined ||
        first_rank === undefined ||
        first_total === undefined ||
        first_percentile === undefined
    ) {
        firstResultBox.innerHTML = "<p>첫 번째 평가 결과를 불러올 수 없습니다.</p>";
        return;
    }

    firstResultBox.innerHTML = `
        <p><strong>First Fluency Score:</strong> ${first_score.toFixed(3)}</p>
        <p><strong>Logic Score:</strong> ${logic_score.toFixed(3)}</p>
        <p><strong>Stutter:</strong> ${stutter_count} / <strong>Filler:</strong> ${filler_count}</p>
        <p><strong>Rank:</strong> ${first_rank} / ${first_total} (상위 ${first_percentile.toFixed(1)}%)</p>
    `;
}

function renderLeaderboardFirst(list, myScore, myName) {
    leaderboardFirstBody.innerHTML = "";
    if (!list || list.length === 0) {
        leaderboardFirstBody.innerHTML = `<tr><td colspan="6">데이터가 없습니다.</td></tr>`;
        return;
    }
    list.forEach((row, idx) => {
        const tr = document.createElement("tr");
        if (row.username === myName && Math.abs(row.score - myScore) < 1e-6) {
            tr.style.backgroundColor = "#fff7d6";
        }
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td>${row.username}</td>
            <td>${row.score.toFixed(3)}</td>
            <td>${row.category}</td>
            <td>${row.keyword}</td>
            <td>${row.created_at}</td>
        `;
        leaderboardFirstBody.appendChild(tr);
    });
}

// ===================================================
// 5. 두 번째 발화 녹음/업로드
// ===================================================
btnStartRec2.addEventListener("click", async () => {
    if (
        !modelAnswerBox.textContent ||
        modelAnswerBox.textContent.includes("아직 모범 답변이") ||
        modelAnswerBox.textContent.includes("모범 답변이 비어 있습니다.")
    ) {
        recordStatus2.textContent = "먼저 모범 답변을 생성한 뒤 2차 발화를 녹음하세요.";
        return;
    }
    if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
        recordStatus2.textContent = "브라우저에서 마이크를 지원하지 않습니다.";
        return;
    }
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        mediaRecorder2 = new MediaRecorder(stream);
        audioChunks2 = [];

        mediaRecorder2.ondataavailable = (event) => {
            if (event.data.size > 0) audioChunks2.push(event.data);
        };
        mediaRecorder2.onstop = () => {
            recordedBlob2 = new Blob(audioChunks2, { type: "audio/webm" });
            audioChunks2 = [];
            audioPreview2.src = URL.createObjectURL(recordedBlob2);
            audioPreview2.style.display = "block";
            recordStatus2.textContent = "두 번째 발화 녹음이 완료되었습니다.";
        };

        mediaRecorder2.start();
        recordStatus2.textContent = "녹음 중입니다. 모범 답변을 읽어주세요...";
        btnStartRec2.disabled = true;
        btnStopRec2.disabled = false;
        audioPreview2.style.display = "none";
        recordedBlob2 = null;
    } catch (err) {
        console.error(err);
        recordStatus2.textContent = "마이크 사용 중 오류가 발생했습니다.";
    }
});

btnStopRec2.addEventListener("click", () => {
    if (mediaRecorder2 && mediaRecorder2.state === "recording") {
        mediaRecorder2.stop();
        btnStartRec2.disabled = false;
        btnStopRec2.disabled = true;
    }
});

// ===================================================
// 6. 두 번째 평가 요청
// ===================================================
btnSubmitSecond.addEventListener("click", async () => {
    const username = usernameInput.value.trim();
    const keyword = keywordInput.value.trim();
    const categoryLabel = categorySelect.options[categorySelect.selectedIndex].text;

    if (!currentAttemptId) {
        submitStatus2.textContent = "먼저 첫 번째 평가를 완료해야 합니다.";
        return;
    }
    if (!username) {
        submitStatus2.textContent = "이름을 입력하세요.";
        return;
    }

    let audioFileToSend = null;
    if (recordedBlob2) {
        audioFileToSend = new File([recordedBlob2], "second.webm", { type: "audio/webm" });
    } else if (audioFileInput2.files.length > 0) {
        audioFileToSend = audioFileInput2.files[0];
    }
    if (!audioFileToSend) {
        submitStatus2.textContent = "두 번째 발화를 녹음하거나 파일을 업로드하세요.";
        return;
    }

    submitStatus2.textContent = "두 번째 발화를 평가 중입니다...";
    btnSubmitSecond.disabled = true;

    const formData = new FormData();
    formData.append("attempt_id", currentAttemptId);
    formData.append("username", username);
    formData.append("category", categoryLabel);
    formData.append("keyword", keyword);
    formData.append("question", selectedQuestion || "");
    formData.append("model_answer", modelAnswerBox.textContent || "");
    formData.append("audio", audioFileToSend);

    try {
        const res = await fetch("/api/second_evaluate", {
            method: "POST",
            body: formData,
        });
        if (!res.ok) throw new Error("Server error");
        const data = await res.json();

        latestSecondScore = data.second_score;
        renderSecondResult(data);
        renderLeaderboardSecond(data.second_leaderboard, data.second_score, username);

        submitStatus2.textContent = "두 번째 평가가 완료되었습니다.";
    } catch (err) {
        console.error(err);
        submitStatus2.textContent = "두 번째 평가 중 오류가 발생했습니다.";
    } finally {
        btnSubmitSecond.disabled = false;
    }
});

function renderSecondResult(data) {
    const {
        second_score,
        second_rank,
        second_total,
        second_percentile,
        first_score,
        improvement,
        stutter_count,
        filler_count,
    } = data;

    if (
        second_score === undefined ||
        second_rank === undefined ||
        second_total === undefined ||
        second_percentile === undefined
    ) {
        secondResultBox.innerHTML = "<p>두 번째 평가 결과를 불러올 수 없습니다.</p>";
        return;
    }

    secondResultBox.innerHTML = `
        <p><strong>Second Fluency Score:</strong> ${second_score.toFixed(3)}</p>
        <p><strong>Stutter:</strong> ${stutter_count} / <strong>Filler:</strong> ${filler_count}</p>
        <p><strong>Second Rank:</strong> ${second_rank} / ${second_total} (상위 ${second_percentile.toFixed(1)}%)</p>
        <p><strong>First vs Second:</strong> ${first_score.toFixed(3)} → ${second_score.toFixed(3)} (Δ ${improvement.toFixed(3)})</p>
    `;
}

function renderLeaderboardSecond(list, myScore, myName) {
    leaderboardSecondBody.innerHTML = "";
    if (!list || list.length === 0) {
        leaderboardSecondBody.innerHTML = `<tr><td colspan="6">데이터가 없습니다.</td></tr>`;
        return;
    }
    list.forEach((row, idx) => {
        const tr = document.createElement("tr");
        if (row.username === myName && Math.abs(row.score - myScore) < 1e-6) {
            tr.style.backgroundColor = "#fff7d6";
        }
        tr.innerHTML = `
            <td>${idx + 1}</td>
            <td>${row.username}</td>
            <td>${row.score.toFixed(3)}</td>
            <td>${row.category}</td>
            <td>${row.keyword}</td>
            <td>${row.created_at}</td>
        `;
        leaderboardSecondBody.appendChild(tr);
    });
}
