import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile, FileBlob} from '@oai/artifact-tool';
const root=process.cwd();
const taskDir=path.join(root,'paper/manuscript/revisions/2026-10-08_minor_revisions_numbering');
const tasks=JSON.parse(await fs.readFile(path.join(taskDir,'xlsx_tasks.json'),'utf8'));
if(process.argv.includes('--help-rename')) {
 const wb=Workbook.create();
 console.log(wb.help('worksheet.name',{include:'index,examples,notes',maxChars:2500}).ndjson);
 process.exit(0);
}
const preview=path.join(root,'paper/tables/_preview_main_tables');
await fs.mkdir(preview,{recursive:true});
const col=n=>String.fromCharCode(64+n);
let completed=0;
for(const item of tasks.new) {
 const wb=Workbook.create();const s=wb.worksheets.add(`Table ${item.table}`);
 s.showGridLines=false;
 const cols=item.rows[0].length;const end=col(cols);const last=item.rows.length+3;
 s.getRange(`A1:${end}1`).merge();s.getRange('A1').values=[[item.caption]];
 s.getRange(`A1:${end}${last+2}`).format.font={name:'Times New Roman',size:11,color:'#000000'};
 s.getRange(`A1:${end}1`).format.font={bold:true,size:14};s.getRange(`A1:${end}1`).format.rowHeight=40;
 s.getRange(`A1:${end}1`).format.wrapText=true;
 s.getRange(`A4:${end}${last}`).values=item.rows;
 s.getRange(`A4:${end}${last}`).format.wrapText=true;
 s.getRange(`A4:${end}${last}`).format.verticalAlignment='center';
 s.getRange(`A4:${end}4`).format={font:{bold:true},rowHeight:45,
   borders:{top:{style:'thin',color:'#000000'},bottom:{style:'thin',color:'#000000'}}};
 s.getRange(`A${last}:${end}${last}`).format.borders={bottom:{style:'thin',color:'#000000'}};
 const widths=item.table===2?[32,70,66]:item.table===3?[34,28,18,48,48,57,20]:item.table===4?[25,28,22,22,25,35]:[65,25,35];
 widths.forEach((v,i)=>s.getRange(`${col(i+1)}1:${col(i+1)}${last+2}`).format.columnWidth=v);
 s.getRange(`A5:${end}${last}`).format.rowHeight=item.table===3?92:item.table===2?82:40;
 s.getRange(`A${last+2}:${end}${last+2}`).merge();s.getRange(`A${last+2}`).values=[['Source: current manuscript table. Existing values copied; no new experimental analysis.']];
 s.getRange(`A${last+2}:${end}${last+2}`).format={wrapText:true,font:{italic:true,size:10},rowHeight:35};
 await (await SpreadsheetFile.exportXlsx(wb)).save(item.path);
 const img=await wb.render({sheetName:s.name,range:`A1:${end}${last+2}`,scale:1.5,format:'png'});
 await fs.writeFile(path.join(preview,`Table_${item.table}.png`),new Uint8Array(await img.arrayBuffer()));
 console.log(`created ${++completed}/${tasks.new.length+tasks.edit.length}: Table ${item.table}`);
}
for(const item of tasks.edit) {
 const wb=await SpreadsheetFile.importXlsx(await FileBlob.load(item.path));
 const beforeSheet=Object.keys(item.rename_sheets)[0]||item.changes[0]?.sheet;
 if(beforeSheet) {
  const s=wb.worksheets.getItem(beforeSheet);
  const old=await wb.render({sheetName:beforeSheet,range:'A1:D10',scale:1,format:'png'});
  await fs.writeFile(path.join(taskDir,`before-${path.basename(item.path)}.png`),new Uint8Array(await old.arrayBuffer()));
 }
 for(const c of item.changes)wb.worksheets.getItem(c.sheet).getRange(c.cell).values=[[c.value]];
 // Worksheet rename is absent from the documented API; apply only that metadata
 // through a separate deterministic workbook.xml patch after export.
 wb.recalculate();
 await (await SpreadsheetFile.exportXlsx(wb)).save(item.path);
 const s=wb.worksheets.getItemAt(0);
 const num=/[\\/]table(\d+)[\\/]/.exec(item.path)?.[1];
 if(num) {
  const ranges={'1':'A1:D10','6':'A1:G18','7':'A1:I20','8':'A1:C18'};
  const img=await wb.render({sheetName:s.name,range:ranges[num]||'A1:G20',scale:1.5,format:'png'});
  await fs.writeFile(path.join(preview,`Table_${num}.png`),new Uint8Array(await img.arrayBuffer()));
 }
 console.log(`edited ${++completed}/${tasks.new.length+tasks.edit.length}: ${path.basename(item.path)}`);
}
console.log('TABLE_ASSETS_RENUMBERED');
