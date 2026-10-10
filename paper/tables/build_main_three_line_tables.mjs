import fs from "node:fs/promises";
import path from "node:path";
import { FileBlob, SpreadsheetFile, Workbook } from "@oai/artifact-tool";

const ROOT = process.cwd();
const FONT = "Times New Roman";
const BLACK = "#000000";
const GREY = "#666666";
const PREVIEW_DIR = path.join(ROOT, "paper", "tables", "_preview_main_tables");

function colName(n) {
  let s = "";
  while (n > 0) {
    n -= 1;
    s = String.fromCharCode(65 + (n % 26)) + s;
    n = Math.floor(n / 26);
  }
  return s;
}

function parseCsv(text) {
  const rows = [];
  let row = [];
  let field = "";
  let quoted = false;
  for (let i = 0; i < text.length; i += 1) {
    const ch = text[i];
    if (quoted) {
      if (ch === '"' && text[i + 1] === '"') {
        field += '"';
        i += 1;
      } else if (ch === '"') {
        quoted = false;
      } else {
        field += ch;
      }
    } else if (ch === '"') {
      quoted = true;
    } else if (ch === ",") {
      row.push(field);
      field = "";
    } else if (ch === "\n") {
      row.push(field.replace(/\r$/, ""));
      rows.push(row);
      row = [];
      field = "";
    } else {
      field += ch;
    }
  }
  if (field.length || row.length) {
    row.push(field.replace(/\r$/, ""));
    rows.push(row);
  }
  return rows;
}

function baseSheet(workbook, name, title, subtitle, nCols) {
  const sheet = workbook.worksheets.add(name);
  sheet.showGridLines = false;
  const lastCol = colName(nCols);
  sheet.mergeCells(`A1:${lastCol}1`);
  sheet.mergeCells(`A2:${lastCol}2`);
  sheet.getRange("A1").values = [[title]];
  sheet.getRange("A2").values = [[subtitle]];
  sheet.getRange(`A1:${lastCol}1`).format = {
    font: { name: FONT, size: 12, bold: true, color: BLACK },
    horizontalAlignment: "left",
    verticalAlignment: "center",
  };
  sheet.getRange(`A2:${lastCol}2`).format = {
    font: { name: FONT, size: 9, italic: true, color: GREY },
    horizontalAlignment: "left",
    verticalAlignment: "center",
    wrapText: true,
  };
  sheet.getRange("A1").format.rowHeight = 22;
  sheet.getRange("A2").format.rowHeight = 30;
  return sheet;
}

function styleThreeLineTable(sheet, headerRow, lastRow, nCols) {
  const lastCol = colName(nCols);
  const all = sheet.getRange(`A${headerRow}:${lastCol}${lastRow}`);
  all.format.borders = { preset: "none" };
  all.format.font = { name: FONT, size: 9, color: BLACK };
  all.format.verticalAlignment = "center";
  const header = sheet.getRange(`A${headerRow}:${lastCol}${headerRow}`);
  header.format.font = { name: FONT, size: 9, bold: true, color: BLACK };
  header.format.horizontalAlignment = "center";
  header.format.verticalAlignment = "center";
  header.format.wrapText = true;
  header.format.rowHeight = 34;
  header.format.borders = {
    top: { style: "medium", color: BLACK },
    bottom: { style: "thin", color: BLACK },
  };
  sheet.getRange(`A${lastRow}:${lastCol}${lastRow}`).format.borders = {
    bottom: { style: "medium", color: BLACK },
  };
}

async function finish(workbook, outputPath, previewName, inspectRange) {
  workbook.recalculate();
  const check = await workbook.inspect({
    kind: "table",
    range: inspectRange,
    include: "values,formulas",
    tableMaxRows: 30,
    tableMaxCols: 12,
    maxChars: 8000,
  });
  console.log(check.ndjson);
  const errors = await workbook.inspect({
    kind: "match",
    searchTerm: "#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!",
    options: { useRegex: true, maxResults: 100 },
    summary: "formula error scan",
  });
  console.log(errors.ndjson);
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  const preview = await workbook.render({ sheetName: "Table", autoCrop: "all", scale: 1.5, format: "png" });
  await fs.writeFile(path.join(PREVIEW_DIR, previewName), new Uint8Array(await preview.arrayBuffer()));
  const blob = await SpreadsheetFile.exportXlsx(workbook);
  await blob.save(outputPath);
  const saved = await SpreadsheetFile.importXlsx(await FileBlob.load(outputPath));
  const savedCheck = await saved.inspect({
    kind: "table",
    range: inspectRange,
    include: "values,formulas",
    tableMaxRows: 30,
    tableMaxCols: 12,
    maxChars: 8000,
  });
  if (!savedCheck.ndjson.includes('"kind":"table"')) {
    throw new Error(`Saved-workbook verification failed: ${outputPath}`);
  }
}

async function buildTable1() {
  const source = JSON.parse(await fs.readFile(path.join(ROOT,"paper","tables","table1","table1_source.json"),"utf8"));
  const workbook = Workbook.create();
  const sheet = baseSheet(workbook,"Table",source.caption,"Full reference matrix: Supplementary Table S1; development method comparison: Table 7.",4);
  const last = 4 + source.rows.length;
  sheet.getRange("A4:D4").values = [source.columns];
  sheet.getRange(`A5:D${last}`).values = source.rows;
  styleThreeLineTable(sheet,4,last,4);
  sheet.getRange(`A5:D${last}`).format.wrapText = true;
  sheet.getRange(`A5:D${last}`).format.rowHeight = 80;
  [32,34,54,58].forEach((width,i)=>sheet.getRange(`${colName(i+1)}1:${colName(i+1)}${last+2}`).format.columnWidth=width);
  sheet.getRange(`A${last+2}:D${last+2}`).merge();
  sheet.getRange(`A${last+2}`).values = [[source.note]];
  sheet.getRange(`A${last+2}:D${last+2}`).format = {font:{name:FONT,size:9,italic:true,color:GREY},wrapText:true,rowHeight:45};
  const out=path.join(ROOT,"paper","tables","table1","Table_1_Related_Work.xlsx");
  await finish(workbook,out,"Table_1.png",`Table!A1:D${last+2}`);
  return out;
}

async function buildTable6() {
  const workbook = Workbook.create();
  const sheet = baseSheet(
    workbook,
    "Table",
    "Table 6. AC-Twin audit across the three measured scopes",
    "Independent confirmation sets; n = 2,000 ten-action decision groups per scope. Lower AURC-regret and Regret@80 are better; higher PR−prevalence is better.",
    7,
  );
  const headers = [["Scope", "Method", "AURC-regret ↓", "Regret@80 ↓", "PR−prevalence ↑", "ΔAURC-regret [95% CI]", "ΔRegret@80 [95% CI]"]];
  const rows = [
    ["Merge ID", "AC-Twin audit", 0.010783, 0.014346, 0.314627, "0.4392 [0.3001, 0.5044]", "0.2600 [0.1695, 0.3230]"],
    [null, "SEED_VOTE_ENTROPY (frozen primary)", 0.019228, 0.019388, 0.054195, null, null],
    [null, "MARGIN_ONLY", 0.018671, 0.020285, 0.054327, null, null],
    [null, "SEED_MARGIN_STD", 0.025405, 0.022550, -0.011707, null, null],
    ["JOINT_OOD", "AC-Twin audit", 0.024473, 0.026289, 0.215680, "0.2077 [0.1276, 0.3050]", "0.1855 [0.1274, 0.2366]"],
    [null, "SEED_VOTE_ENTROPY (frozen primary)", 0.030887, 0.032275, 0.046445, null, null],
    [null, "MARGIN_ONLY", 0.030431, 0.032628, 0.062315, null, null],
    [null, "SEED_MARGIN_STD", 0.036474, 0.035893, -0.025396, null, null],
    ["MetaDrive", "AC-Twin audit", 0.030011, 0.045276, 0.211717, "0.1949 [0.1281, 0.2575]", "0.1532 [0.1070, 0.1975]"],
    [null, "SEED_MARGIN_STD (frozen primary)", 0.037276, 0.053470, 0.075112, null, null],
    [null, "MARGIN_ONLY", 0.065541, 0.062677, 0.008458, null, null],
    [null, "SEED_VOTE_ENTROPY", 0.039973, 0.050337, 0.126450, null, null],
  ];
  sheet.getRange("A4:G4").values = headers;
  sheet.getRange("A5:G16").values = rows;
  styleThreeLineTable(sheet, 4, 16, 7);
  sheet.mergeCells("A5:A8");
  sheet.mergeCells("A9:A12");
  sheet.mergeCells("A13:A16");
  sheet.getRange("A5:A16").format = { font: { name: FONT, size: 9, bold: true, color: BLACK }, horizontalAlignment: "center", verticalAlignment: "center", wrapText: true };
  sheet.getRange("B5:B16").format.wrapText = true;
  sheet.getRange("C5:E16").format.numberFormat = "0.000000";
  sheet.getRange("C5:G16").format.horizontalAlignment = "right";
  for (const row of [5, 9, 13]) {
    sheet.getRange(`B${row}:G${row}`).format.font = { name: FONT, size: 9, bold: true, color: BLACK };
  }
  for (const row of [6, 10, 14]) {
    sheet.getRange(`B${row}:G${row}`).format.font = { name: FONT, size: 9, italic: true, color: BLACK };
  }
  sheet.getRange("A5:G16").format.rowHeight = 24;
  const widths = [17, 35, 16, 16, 17, 27, 27];
  widths.forEach((w, i) => { sheet.getRange(`${colName(i + 1)}1:${colName(i + 1)}16`).format.columnWidth = w; });
  sheet.mergeCells("A18:G18");
  sheet.getRange("A18").values = [["Note: Δ values are relative improvements against the validation-frozen primary baseline within each scope. The three scopes differ in task distribution, baseline and model relation and must not be ranked or pooled across rows."]];
  sheet.getRange("A18:G18").format = { font: { name: FONT, size: 8, italic: true, color: GREY }, wrapText: true };
  sheet.getRange("A18").format.rowHeight = 34;
  const out = path.join(ROOT, "paper", "tables", "table6", "Table_6_Confirmed_Results.xlsx");
  await finish(workbook, out, "Table_6.png", "Table!A1:G18");
  return out;
}

async function buildTable7() {
  const csvPath = path.join(ROOT, "paper", "tables", "table7", "table7_comparison_main.csv");
  const csvRows = parseCsv(await fs.readFile(csvPath, "utf8"));
  csvRows[0][0] = csvRows[0][0].replace(/^\uFEFF/, "");
  const headersMap = Object.fromEntries(csvRows[0].map((x, i) => [x, i]));
  const records = csvRows.slice(1).map((r) => ({
    category: r[headersMap.category],
    method: r[headersMap.method],
    venue: r[headersMap.venue],
    aurc: Number(r[headersMap.aurc_regret]),
    r80: Number(r[headersMap.regret_at_80]),
    pr: Number(r[headersMap.pr_auc_minus_prevalence]),
    spearman: Number(r[headersMap.spearman]),
    params: r[headersMap.param_count_per_seed] ? Number(r[headersMap.param_count_per_seed]) : null,
    flops: r[headersMap.approx_forward_flops_per_group] ? Number(r[headersMap.approx_forward_flops_per_group]) : null,
    ms: r[headersMap.inference_ms_per_group] ? Number(r[headersMap.inference_ms_per_group]) : null,
  }));
  const workbook = Workbook.create();
  const sheet = baseSheet(
    workbook,
    "Table",
    "Table 7. Comparison with adapted published methods on the shared development benchmark",
    "ADAPTIVE_DEVELOPMENT_ONLY (Stage 4C-R2 validation, 1,500 groups); this table is a shared development-benchmark comparison, not a confirmatory SOTA-superiority claim.",
    9,
  );
  const headers = [["Method", "Venue", "AURC-regret ↓", "Regret@80 ↓", "PR-AUC − prevalence ↑", "Spearman ↑", "Params/seed", "Approx. FLOPs/group ↓", "Inference (ms/group) ↓"]];
  sheet.getRange("A4:I4").values = headers;
  const groups = [
    ["Adapted published methods", records.filter((r) => r.category === "published")],
    ["Internal baselines", records.filter((r) => r.category === "internal")],
    ["This work", records.filter((r) => r.category === "ours")],
  ];
  let row = 5;
  const dataRows = [];
  const groupRows = [];
  const oursRows = [];
  for (const [label, items] of groups) {
    groupRows.push(row);
    sheet.mergeCells(`A${row}:I${row}`);
    sheet.getRange(`A${row}`).values = [[label]];
    row += 1;
    for (const item of items) {
      const values = [[item.method, item.venue, item.aurc, item.r80, item.pr, item.spearman, item.params, item.flops, item.ms]];
      sheet.getRange(`A${row}:I${row}`).values = values;
      dataRows.push(row);
      if (item.category === "ours") oursRows.push(row);
      row += 1;
    }
  }
  const lastRow = row - 1;
  styleThreeLineTable(sheet, 4, lastRow, 9);
  for (const r of groupRows) {
    sheet.getRange(`A${r}:I${r}`).format = { font: { name: FONT, size: 9, bold: true, italic: true, color: GREY }, horizontalAlignment: "left", verticalAlignment: "center" };
    sheet.getRange(`A${r}`).format.rowHeight = 19;
  }
  for (const r of dataRows) {
    sheet.getRange(`C${r}:F${r}`).format.numberFormat = "0.0000";
    sheet.getRange(`G${r}:H${r}`).format.numberFormat = "#,##0";
    sheet.getRange(`I${r}`).format.numberFormat = "0.0000";
    sheet.getRange(`C${r}:I${r}`).format.horizontalAlignment = "right";
    sheet.getRange(`A${r}:I${r}`).format.rowHeight = 24;
  }
  for (const r of oursRows) {
    sheet.getRange(`A${r}:I${r}`).format.font = { name: FONT, size: 9, bold: true, color: BLACK };
  }
  sheet.getRange(`A${lastRow}:I${lastRow}`).format.borders = { bottom: { style: "medium", color: BLACK } };
  const widths = [38, 32, 16, 16, 22, 14, 15, 21, 20];
  widths.forEach((w, i) => { sheet.getRange(`${colName(i + 1)}1:${colName(i + 1)}${lastRow}`).format.columnWidth = w; });
  sheet.getRange(`A5:B${lastRow}`).format.wrapText = true;
  const noteRow = lastRow + 2;
  sheet.mergeCells(`A${noteRow}:I${noteRow}`);
  sheet.getRange(`A${noteRow}`).values = [["Notes: All learned methods use five seeds. FLOPs are simplified per-seed parameterized-forward estimates (MAC×2). Timing is the median batch-1 five-seed ensemble risk-head latency on an NVIDIA GeForce RTX 3070 Ti; the shared upstream predictor is excluded."]];
  sheet.getRange(`A${noteRow}:I${noteRow}`).format = { font: { name: FONT, size: 8, italic: true, color: GREY }, wrapText: true };
  sheet.getRange(`A${noteRow}`).format.rowHeight = 34;
  const out = path.join(ROOT, "paper", "tables", "table7", "Table_7_Published_Comparison.xlsx");
  await finish(workbook, out, "Table_7.png", `Table!A1:I${noteRow}`);
  return out;
}

const outputs = [];
outputs.push(await buildTable1());
if (!process.argv.includes("--only-table-1") && !process.argv.includes("--only-table-i")) {
  outputs.push(await buildTable6());
  outputs.push(await buildTable7());
}
console.log(JSON.stringify({ outputs }, null, 2));
