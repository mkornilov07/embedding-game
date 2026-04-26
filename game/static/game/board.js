/* Shared word-puzzle board: renders a word bank + equation rows with drag/drop.
 *
 * Usage:
 *   const board = createBoard({
 *     bankEl:        <div>,     // container for word bubbles
 *     boardEl:       <div>,     // container for equation rows (new rows appended)
 *     words:         [{text, combo}, ...],
 *     combinations:  [{id, type}, ...],   // only 'word_sum' (3 slots) for now
 *     onRowChanged:  (comboIdx) => {},    // fired after anything in that row changes
 *   });
 *
 * The callback is fired with the combo index as a string. It fires for every
 * affected row — including both source and target when a drag moves a word
 * between rows, and when a word is returned to the bank.
 *
 * Returned helpers expose the board state so callers can decide what to do
 * (e.g. check a row with the server, or verify all rows are filled).
 */
(function (global) {
  // Delay between marking a row solved and visually moving it to the solved
  // list. Long enough to read the green slot styling, short enough to feel snappy.
  const SOLVE_DELAY_MS = 450;

  // CSS classes assigned by checkRowAndStyle from server feedback.
  const SLOT_FEEDBACK_CLASSES = ["slot-wrong", "slot-correct", "slot-shuffled"];

  function createBoard({ bankEl, boardEl, words, combinations, onRowChanged }) {
    let dragIndex = null;
    let dragSourceBox = null;

    words.forEach((word, i) => {
      const el = document.createElement("div");
      el.className = "word-bubble";
      el.textContent = word.text;
      el.draggable = true;
      el.dataset.index = i;
      el.addEventListener("dragstart", onBubbleDragStart);
      el.addEventListener("dragend", onBubbleDragEnd);
      el.addEventListener("click", onBubbleClick);
      bankEl.appendChild(el);
    });

    combinations.forEach((combo, i) => {
      const row = document.createElement("div");
      row.className = "equation";
      row.dataset.combo = i;
      if (combo.type === "word_sum") {
        row.append(
          makeDropBox(i, 0),
          makeOperator("+"),
          makeDropBox(i, 1),
          makeOperator("="),
          makeDropBox(i, 2),
        );
      }
      boardEl.appendChild(row);
    });

    bankEl.addEventListener("dragover", e => { if (dragSourceBox) e.preventDefault(); });
    bankEl.addEventListener("drop", e => {
      if (!dragSourceBox || dragIndex === null) return;
      e.preventDefault();
      const comboIdx = dragSourceBox.dataset.combo;
      clearBox(dragSourceBox);
      markBubblePlaced(dragIndex, false);
      fire(comboIdx);
    });

    function makeOperator(text) {
      const op = document.createElement("span");
      op.className = "operator";
      op.textContent = text;
      return op;
    }

    function makeDropBox(comboIdx, slot) {
      const box = document.createElement("div");
      box.className = "drop-box";
      box.dataset.combo = comboIdx;
      box.dataset.slot = slot;
      box.draggable = true;
      box.addEventListener("dragstart", onBoxDragStart);
      box.addEventListener("dragend", onBoxDragEnd);
      box.addEventListener("dragover", onDragOver);
      box.addEventListener("dragleave", onDragLeave);
      box.addEventListener("drop", onDrop);
      box.addEventListener("click", onBoxClick);
      return box;
    }

    function onBubbleDragStart(e) {
      const b = e.target;
      if (b.classList.contains("placed")) { e.preventDefault(); return; }
      dragIndex = b.dataset.index;
      dragSourceBox = null;
      b.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
    }
    function onBubbleDragEnd(e) {
      e.target.classList.remove("dragging");
      cleanupDrag();
    }

    function onBoxDragStart(e) {
      const box = e.currentTarget;
      if (box.dataset.wordIndex === undefined) { e.preventDefault(); return; }
      dragIndex = box.dataset.wordIndex;
      dragSourceBox = box;
      box.classList.add("dragging");
      e.dataTransfer.effectAllowed = "move";
    }
    function onBoxDragEnd(e) {
      e.currentTarget.classList.remove("dragging");
      cleanupDrag();
    }

    function onDragOver(e) { e.preventDefault(); e.currentTarget.classList.add("drag-over"); }
    function onDragLeave(e) { e.currentTarget.classList.remove("drag-over"); }

    function onDrop(e) {
      e.preventDefault();
      const box = e.currentTarget;
      box.classList.remove("drag-over");
      if (dragIndex === null) return;
      if (box === dragSourceBox) return;

      const targetIdx = box.dataset.wordIndex;
      const sourceCombo = dragSourceBox ? dragSourceBox.dataset.combo : null;

      if (dragSourceBox) {
        // Box-to-box: swap if target filled, else just empty the source.
        if (targetIdx !== undefined) setBoxWord(dragSourceBox, parseInt(targetIdx));
        else clearBox(dragSourceBox);
      } else {
        // Bank-to-box: bounce the occupying word back to the bank first.
        if (targetIdx !== undefined) markBubblePlaced(targetIdx, false);
        markBubblePlaced(dragIndex, true);
      }
      setBoxWord(box, parseInt(dragIndex));

      if (sourceCombo !== null && sourceCombo !== box.dataset.combo) {
        fire(sourceCombo);
      }
      fire(box.dataset.combo);
    }

    function onBubbleClick(e) {
      const b = e.target;
      if (b.classList.contains("placed")) return;
      const empty = boardEl.querySelector(".drop-box:not(.filled)");
      if (!empty) return;
      b.classList.add("placed");
      setBoxWord(empty, parseInt(b.dataset.index));
      fire(empty.dataset.combo);
    }

    function onBoxClick(e) {
      const box = e.currentTarget;
      if (box.dataset.wordIndex === undefined) return;
      const comboIdx = box.dataset.combo;
      markBubblePlaced(box.dataset.wordIndex, false);
      clearBox(box);
      fire(comboIdx);
    }

    function setBoxWord(box, wordIdx) {
      box.textContent = words[wordIdx].text;
      box.dataset.wordIndex = wordIdx;
      box.classList.add("filled");
      box.classList.remove(...SLOT_FEEDBACK_CLASSES);
    }
    function clearBox(box) {
      box.textContent = "";
      delete box.dataset.wordIndex;
      box.classList.remove("filled", ...SLOT_FEEDBACK_CLASSES);
    }
    function markBubblePlaced(wordIdx, placed) {
      const b = bankEl.querySelector(`[data-index="${wordIdx}"]`);
      if (b) b.classList.toggle("placed", placed);
    }
    function cleanupDrag() { dragIndex = null; dragSourceBox = null; }
    function fire(comboIdx) {
      if (onRowChanged) onRowChanged(String(comboIdx));
    }

    return {
      boxes(comboIdx) {
        return boardEl.querySelectorAll(`.drop-box[data-combo="${comboIdx}"]`);
      },
      isRowFilled(comboIdx) {
        return Array.from(this.boxes(comboIdx))
          .every(b => b.dataset.wordIndex !== undefined);
      },
      rowWords(comboIdx) {
        return Array.from(this.boxes(comboIdx))
          .map(b => words[parseInt(b.dataset.wordIndex)].text);
      },
      allRowsFilled() {
        return boardEl.querySelectorAll(".drop-box:not(.filled)").length === 0;
      },
      eachCombo(fn) {
        combinations.forEach((_, i) => fn(String(i)));
      },
    };
  }

  /* Collapsible "N solved ✓" group at the top of the board.
   *
   * Usage:
   *   const solved = createSolvedGroup({
   *     sectionEl: <div>,      // outer group wrapper (matches puzzle.html markup)
   *     listEl:    <div>,      // inner list that holds the moved rows
   *     countEl:   <span>,     // text node for the counter
   *     toggleEl:  <button>,   // collapse/expand button
   *     boardEl:   <div>,      // main equations container (where rows live when not solved)
   *   });
   *
   *   solved.markSolved(comboIdx);   // schedule the row to move into the solved list
   *   solved.markUnsolved(comboIdx); // move it back immediately; cancels any pending solve
   *   solved.count();                // logical solved count (pending + already moved)
   */
  function createSolvedGroup({ sectionEl, listEl, countEl, toggleEl, boardEl }) {
    // Pending rows are scheduled to move into the solved list after a delay,
    // but already count as solved so completion checks aren't blocked by it.
    const pending = new Map();

    toggleEl.addEventListener("click", () => {
      const collapsed = sectionEl.classList.toggle("collapsed");
      toggleEl.setAttribute("aria-expanded", String(!collapsed));
    });

    function cancelPending(comboIdx) {
      const t = pending.get(comboIdx);
      if (t !== undefined) { clearTimeout(t); pending.delete(comboIdx); }
    }

    function rowFor(comboIdx) {
      return document.querySelector(`.equation[data-combo="${comboIdx}"]`);
    }

    function markSolved(comboIdx) {
      const row = rowFor(comboIdx);
      if (!row) return;
      cancelPending(comboIdx);
      const t = setTimeout(() => {
        pending.delete(comboIdx);
        if (row.parentElement !== listEl) {
          listEl.appendChild(row);
          update();
        }
      }, SOLVE_DELAY_MS);
      pending.set(comboIdx, t);
    }

    function markUnsolved(comboIdx) {
      cancelPending(comboIdx);
      const row = rowFor(comboIdx);
      if (!row || row.parentElement !== listEl) return;
      const idx = parseInt(comboIdx);
      const siblings = boardEl.querySelectorAll(":scope > .equation");
      let inserted = false;
      for (const sib of siblings) {
        if (parseInt(sib.dataset.combo) > idx) {
          boardEl.insertBefore(row, sib);
          inserted = true;
          break;
        }
      }
      if (!inserted) boardEl.appendChild(row);
      update();
    }

    function update() {
      const n = listEl.children.length;
      sectionEl.hidden = n === 0;
      if (n > 0) countEl.textContent = `${n} solved ✓`;
    }

    return {
      markSolved, markUnsolved,
      count: () => pending.size + listEl.children.length,
    };
  }

  /* Reset a single row's slot styling and unmark it from the solved group.
   * Call at the start of any row change to wipe stale check feedback. */
  function clearRowVisuals({ board, solved, comboIdx }) {
    board.boxes(comboIdx).forEach(b => b.classList.remove(...SLOT_FEEDBACK_CLASSES));
    solved.markUnsolved(comboIdx);
  }

  /* Run error-check on a single row: POST its words, then style each slot
   * (green/yellow/red) and drive the solved group. Returns a promise that
   * resolves to true iff every slot is green.
   *
   * The server endpoint is expected to return
   *   { green_slots: [indices], yellow_slots: [indices] }.
   * Slots not in either list render red.
   */
  async function checkRowAndStyle({ url, csrfToken, board, solved, comboIdx }) {
    const slotWords = board.rowWords(comboIdx);
    const boxes = board.boxes(comboIdx);
    const r = await fetch(url, {
      method: "POST",
      headers: {"Content-Type": "application/json", "X-CSRFToken": csrfToken},
      body: JSON.stringify({ words: slotWords }),
    });
    const data = await r.json();
    const greenSet = new Set(data.green_slots || []);
    const yellowSet = new Set(data.yellow_slots || []);
    boxes.forEach((box, slotIdx) => {
      box.classList.remove(...SLOT_FEEDBACK_CLASSES);
      if (greenSet.has(slotIdx)) box.classList.add("slot-correct");
      else if (yellowSet.has(slotIdx)) box.classList.add("slot-shuffled");
      else box.classList.add("slot-wrong");
    });
    const correct = greenSet.size === 3;
    if (correct) solved.markSolved(comboIdx);
    else solved.markUnsolved(comboIdx);
    return correct;
  }

  /* Toggle the "all filled but wrong arrangement" hint: yellow glow on every
   * drop-box. Already-correct (green) boxes keep their styling — the hint is
   * for boxes that haven't been verified yet. */
  function updateAllFilledHint({ board, solved, expected }) {
    const allFilled = board.allRowsFilled();
    const allDone = solved.count() >= expected;
    const hint = allFilled && !allDone;
    document.querySelectorAll(".drop-box").forEach(b => {
      b.classList.toggle("slot-yellow", hint && !b.classList.contains("slot-correct"));
    });
  }

  function launchFireworks(containerId) {
    const container = document.getElementById(containerId);
    if (!container) return;
    const colors = ["#fbbf24", "#f87171", "#34d399", "#60a5fa", "#a78bfa", "#f472b6"];
    for (let burst = 0; burst < 5; burst++) {
      const cx = Math.random() * window.innerWidth;
      const cy = Math.random() * window.innerHeight * 0.6;
      for (let i = 0; i < 30; i++) {
        const spark = document.createElement("div");
        spark.className = "spark";
        const angle = (Math.PI * 2 * i) / 30;
        const dist = 80 + Math.random() * 120;
        spark.style.left = cx + "px";
        spark.style.top = cy + "px";
        spark.style.background = colors[Math.floor(Math.random() * colors.length)];
        spark.style.setProperty("--dx", Math.cos(angle) * dist + "px");
        spark.style.setProperty("--dy", Math.sin(angle) * dist + "px");
        spark.style.animationDelay = burst * 0.2 + "s";
        container.appendChild(spark);
      }
    }
  }

  global.createBoard = createBoard;
  global.createSolvedGroup = createSolvedGroup;
  global.clearRowVisuals = clearRowVisuals;
  global.checkRowAndStyle = checkRowAndStyle;
  global.updateAllFilledHint = updateAllFilledHint;
  global.launchFireworks = launchFireworks;
})(window);
