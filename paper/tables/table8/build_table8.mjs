import fs from 'node:fs/promises';
import path from 'node:path';
import { fileURLToPath } from 'node:url';
import crypto from 'node:crypto';
import assert from 'node:assert/strict';
import { Workbook, SpreadsheetFile } from '@oai/artifact-tool';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(HERE, '../../..');
const EXP = path.join(ROOT, 'experiments/paper_revision/m1_supervised_ablation_v02');
const MD = path.join(ROOT, 'paper/manuscript/AC-Twin_论文初稿_中文.md');
const WORD = path.join(ROOT, 'paper/manuscript/AC-Twin_论文初稿_中文_Word版.docx');
const hash = b => crypto.createHash('sha256').update(b).digest('hex');
const protectedPaths = [MD, WORD, path.join(EXP,'evaluation/contrasts.csv'),path.join(EXP,'evaluation/bootstrap_receipt.json')];
const before = Object.fromEntries(await Promise.all(protectedPaths.map(async p=>[p,hash(await fs.readFile(p))])));
const manuscript = (await fs.readFile(MD,'utf8')).replace(/\r\n/g,'\n');
const section = manuscript.split('### 6.5 Supervised controls and feature ablations on development data')[1].split('## 7.')[0];
const tableZh = section.match(/^\|.*\|\n(?:\|.*\|\n?)+/m)[0].trim();
const noteZh = section.split('\n').find(line=>line.startsWith('表注：'));
const receipt = JSON.parse(await fs.readFile(path.join(ROOT,'paper/manuscript/revisions/2026-10-08_supervised_controls/integration_receipt.json'),'utf8'));
receipt.caption_en = 'Table 8 | Supervised controls and feature ablations on development data.';
const sourceText = await fs.readFile(path.join(EXP,'evaluation/contrasts.csv'),'utf8');
const csvBook = await Workbook.fromCSV(sourceText,{sheetName:'Import'});
const raw = csvBook.worksheets.getItem('Import').getUsedRange().values;
const header = raw[0];
const records = raw.slice(1).filter(r=>r[0]).map(r=>Object.fromEntries(header.map((key,i)=>[key,r[i]])));
const selected = [];
for(let i=1;i<=5;i++) for(const metric of ['aurc_regret','regret_at_80']){
  const hits=records.filter(r=>r.contrast===`C${i}` && r.metric===metric && r.form==='improvement');
  assert.equal(hits.length,1);
  const r=hits[0]; assert.equal(r.status,'OK'); assert.equal(Number(r.invalid_draws),0);
  selected.push({contrast:r.contrast,metric:r.metric,method:r.method,reference:r.reference,
    point:Number(r.point),ci99_lower:Number(r.ci99_lower),ci99_upper:Number(r.ci99_upper)});
}
const display = r=>`${(100*r.point).toFixed(1)} [${(100*r.ci99_lower).toFixed(1)}, ${(100*r.ci99_upper).toFixed(1)}]`;
for(let i=0;i<5;i++){
  assert.equal(display(selected[2*i]),receipt.word_rows[i][1]);
  assert.equal(display(selected[2*i+1]),receipt.word_rows[i][2]);
  assert(tableZh.includes(`| ${receipt.table_rows[i].join(' | ')} |`));
}
const metadata={table:'Table 8',title:'Supervised controls and feature ablations on development data',
  generated_date:'2026-10-08',analysis_status:'ADAPTIVE_DEVELOPMENT_ONLY',
  independent_unit:'complete ten-action decision group',validation_groups:1500,model_seeds:5,
  bootstrap_draws:10000,interval_level:0.99,interval_quantiles:[0.005,0.995],
  family_size:5,families:['AURC-regret (primary)','Regret@80 (secondary)'],
  family_coverage:'Nominal 95% Bonferroni within each family; no joint 95% across families',
  unit:'dimensionless fraction; manuscript displays 100 times these values as percent',
  source:'experiments/paper_revision/m1_supervised_ablation_v02/evaluation/contrasts.csv',
  source_sha256:before[path.join(EXP,'evaluation/contrasts.csv')],
  input_arms:[{name:'Complete input',active_dimensions:315,nominal_parameters:15740},
    {name:'Baseline signals only',active_dimensions:3,nominal_parameters:15740},
    {name:'Explicit risk channels removed',active_dimensions:180,nominal_parameters:15740}],
  note_zh:noteZh,note_en:receipt.note_en,display_rows:receipt.word_rows,contrasts:selected};
await fs.writeFile(path.join(HERE,'table8_source.json'),JSON.stringify(metadata,null,2));
const csvFields=['contrast','metric','method','reference','point','ci99_lower','ci99_upper'];
await fs.writeFile(path.join(HERE,'table8_data.csv'),csvFields.join(',')+'\n'+selected.map(r=>csvFields.map(k=>r[k]).join(',')).join('\n')+'\n');
const tableEn=['| Comparison (method / reference) | AURC-regret reduction [99% CI] (%) | Regret@80 reduction [99% CI] (%) |','| --- | ---: | ---: |',...receipt.word_rows.map(r=>`| ${r.join(' | ')} |`)].join('\n');
await fs.writeFile(path.join(HERE,'TABLE_8.md'),`# Table 8 — Supervised controls and feature ablations on development data\n\n对应正文§5.5、§6.5及补充方法SM11。五项对照均来自同一1,500组开发验证数据。\n\n## 中文表体（与正文一致）\n\n**Table 8 | 开发范围的监督对照与特征消融。**\n\n${tableZh}\n\n${noteZh}\n\n## English table and caption\n\n**${receipt.caption_en}**\n\n${tableEn}\n\n${receipt.note_en}\n\n## 比较口径\n\n相对改善为1−方法指标/参照指标，正值表示指标降低；机读数据保留原始无量纲比例，正文以百分数显示并四舍五入至一位小数。五项主要AURC-regret与五项次Regret@80分别组成家族，每项99%百分位区间对应其五比较家族Bonferroni名义95%覆盖，不合并声称两个家族整体95%覆盖。开发验证组曾参与历史开发决策，结果不构成新确认。\n\n完整输入相对三标量监督组合有条件追加收益；三标量监督组合相对三项固定启发式的校正区间均跨零。移除显式风险通道的对比两项区间均跨零，不建立通道的独立增量，也不证明等效或风险信息无用。名义参数量相同不等于有效容量相同。\n\n## 数字来源\n\n- \`experiments/paper_revision/m1_supervised_ablation_v02/evaluation/contrasts.csv\`：C1–C5的improvement行，分别提取aurc_regret和regret_at_80的point/ci99_lower/ci99_upper。\n- \`evaluation/bootstrap_receipt.json\`：重采样次数、区间分位与家族口径。\n- \`config.json\`及\`models/model_inventory.json\`：输入方案、训练与参数合同。\n- \`paper/manuscript/AC-Twin_论文初稿_中文.md\`§6.5：中文表体与表注。\n\n本包只整理冻结结果，不训练、推理、重采样或重判。\n`);
await fs.writeFile(path.join(HERE,'README.md'),`# Table 8｜监督对照与特征消融\n\n正文表格编号为Table 8，不使用内部评审问题编号。\n\n| 文件 | 说明 |\n| --- | --- |\n| [TABLE_8.md](TABLE_8.md) | 中英文表体、caption、表注、统计口径及主张边界 |\n| [Table_8_Supervised_Controls.xlsx](Table_8_Supervised_Controls.xlsx) | 可编辑Excel，Table 8为三线表，Data为未四舍五入的十个效应记录 |\n| [table8_data.csv](table8_data.csv) | 机读效应与99%区间，数值单位为无量纲比例 |\n| [table8_source.json](table8_source.json) | 元数据、冻结来源哈希、输入维度、完整数值和显示值 |\n| build_table8.mjs | 只读冻结CSV与正文，重建本包并核对五项对照 |\n\n训练/评价细节见[补充方法SM11](../../manuscript/SUPPLEMENTARY_METHODS.md)。当前正文仍以[唯一中文工作稿](../../manuscript/AC-Twin_论文初稿_中文.md)为权威；本目录为其表格资产包，不维护第二份正文。\n\n本表属于开发性证据。主要AURC和次Regret@80各五对比独立做Bonferroni校正；所有十个99%区间完整保留，包括跨零与负向点估计。未使用未经校正95%区间替代。\n\n重建：从项目根目录运行\`node paper/tables/table8/build_table8.mjs\`，使用已配置的工作区表格依赖；无需训练或读取确认逐组资源。\n`);

const wb=Workbook.create();
const main=wb.worksheets.add('Table 8');
const data=wb.worksheets.add('Data');
for(const s of [main,data]){s.showGridLines=false;s.getRange('A1:G22').format.font={name:'Times New Roman',size:11,color:'#000000'};s.getRange('A1:G22').format.verticalAlignment='center';}
main.getRange('A2:C2').merge(); main.getRange('A2').values=[[receipt.caption_en]]; main.getRange('A2:C2').format.font={size:14,bold:true};main.getRange('A2:C2').format.rowHeight=27;
main.getRange('A4:C4').merge(); main.getRange('A4').values=[['Development data: 1,500 complete ten-action groups. Values in percent.']];main.getRange('A4:C4').format.font={italic:true,size:11};
main.getRange('A6:C6').values=[['Comparison (method / reference)','AURC-regret reduction\n[99% CI] (%)','Regret@80 reduction\n[99% CI] (%)']];
main.getRange('A6:C6').format={font:{bold:true},wrapText:true,horizontalAlignment:'center',rowHeight:36,borders:{top:{style:'thin',color:'#000000'},bottom:{style:'thin',color:'#000000'}}};
main.getRange('A7:A11').values=receipt.word_rows.map(r=>[r[0]]);
main.getRange('A7:C11').format.rowHeight=37;
main.getRange('A7:A11').format.wrapText=true;
main.getRange('B7:C11').format.horizontalAlignment='right';
main.getRange('A11:C11').format.borders={bottom:{style:'thin',color:'#000000'}};
main.getRange('A1:A18').format.columnWidth=63;main.getRange('B1:C18').format.columnWidth=34;
main.getRange('A13:C17').merge();main.getRange('A13').values=[[receipt.note_en]];main.getRange('A13:C17').format.wrapText=true;main.getRange('A13:C17').format.verticalAlignment='top';main.getRange('A13:C17').format.rowHeight=21;

data.getRange('A2:G2').merge();data.getRange('A2').values=[['Relative improvements and 99% intervals (unrounded fractions)']];data.getRange('A2:G2').format.font={size:14,bold:true};data.getRange('A2:G2').format.rowHeight=28;
data.getRange('A4:G4').merge();data.getRange('A4').values=[['Source: '+metadata.source]];data.getRange('A4:G4').format.font={italic:true};data.getRange('A4:G4').format.wrapText=true;data.getRange('A4:G4').format.rowHeight=32;
data.getRange('A6:G6').values=[['Contrast','Metric','Method','Reference','Improvement','CI99 lower','CI99 upper']];
data.getRange('A6:G6').format={font:{bold:true},horizontalAlignment:'center',wrapText:true,rowHeight:30,borders:{top:{style:'thin',color:'#000000'},bottom:{style:'thin',color:'#000000'}}};
data.getRange('A7:G16').values=selected.map(r=>[r.contrast,r.metric,r.method,r.reference,r.point,r.ci99_lower,r.ci99_upper]);
data.getRange('A7:G16').format.rowHeight=22;data.getRange('E7:G16').setNumberFormat('0.0000000000');data.getRange('E7:G16').format.horizontalAlignment='right';
for(const [column,width] of [['A',10],['B',19],['C',25],['D',25],['E',21],['F',21],['G',21]]) data.getRange(`${column}1:${column}19`).format.columnWidth=width;
data.getRange('A16:G16').format.borders={bottom:{style:'thin',color:'#000000'}};
data.getRange('A18:G19').merge();data.getRange('A18').values=[['Each interval is 99%; two separate five-contrast families. Values are fractions, not percentage points. Adaptive development evidence.']];data.getRange('A18:G19').format.wrapText=true;data.getRange('A18:G19').format.verticalAlignment='top';
const composite=row=>`=TEXT('Data'!E${row}*100,"0.0")&" ["&TEXT('Data'!F${row}*100,"0.0")&", "&TEXT('Data'!G${row}*100,"0.0")&"]"`;
for(let i=0;i<5;i++)main.getRange(`B${i+7}:C${i+7}`).formulas=[[composite(7+2*i),composite(8+2*i)]];
// Verify presentation recalculates and restore the frozen source before export.
const original=data.getRange('E7').values;
data.getRange('E7').values=[[0.5]];
assert(main.getRange('B7').values[0][0].startsWith('50.0 '));
data.getRange('E7').values=original;
wb.recalculate();
assert.deepEqual(main.getRange('B7:C11').values,receipt.word_rows.map(r=>r.slice(1)));
const inspection=await wb.inspect({kind:'table',range:'Table 8!A6:C11',include:'values,formulas',tableMaxRows:6,tableMaxCols:3,maxChars:3500});
await fs.writeFile(path.join(HERE,'verification.ndjson'),inspection.ndjson);
const errors=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#N/A|#NUM!|#NULL!|#SPILL!|#CALC!',options:{useRegex:true,maxResults:30},maxChars:1500});
assert(!/(#REF!|#DIV\/0!|#VALUE!|#NAME\?|#NUM!|#SPILL!)/.test(errors.ndjson));
const previewDir=path.join(ROOT,'paper/tables/_preview_main_tables');
await fs.mkdir(previewDir,{recursive:true});
for(const [sheetName,range,file] of [['Table 8','A1:C18','Table_8.png'],['Data','A1:G20','Table_8_Data.png']]){
  const png=await wb.render({sheetName,range,scale:1.5,format:'png'});
  await fs.writeFile(path.join(previewDir,file),new Uint8Array(await png.arrayBuffer()));
}
const output=await SpreadsheetFile.exportXlsx(wb);
await output.save(path.join(HERE,'Table_8_Supervised_Controls.xlsx'));
for(const p of protectedPaths) assert.equal(hash(await fs.readFile(p)),before[p]);
await fs.writeFile(path.join(HERE,'verification.json'),JSON.stringify({table:'Table 8',five_comparisons_verified:true,ten_interval_values_match_source:true,manuscript_display_matches:true,formula_recalculation_verified:true,manuscript_and_word_unchanged:true,scientific_sources_unchanged:true,visual_review:'PENDING'},null,2));
console.log('Table 8 package exported. Five comparisons / ten 99% intervals match frozen sources and manuscript.');
