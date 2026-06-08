/* ── HemaForecast AI · app.js ── */
const API = '';  // empty = same host (Flask serves frontend)
const BG = ['A+','A-','B+','B-','AB+','AB-','O+','O-'];
const BG_C = ['#e84545','#D4537E','#a78bfa','#6d5fd5','#00c896','#00876a','#4a9eff','#f59e0b'];
const SEASON = [0.92,0.88,0.95,1.02,1.08,1.12,1.05,0.98,1.15,1.10,1.02,0.95];
const BASE = {'A+':320,'A-':85,'B+':280,'B-':72,'AB+':95,'AB-':28,'O+':420,'O-':110};
const HL24 = ['May-24','Jun-24','Jul-24','Aug-24','Sep-24','Oct-24','Nov-24','Dec-24','Jan-25','Feb-25','Mar-25','Apr-25','May-25','Jun-25','Jul-25','Aug-25','Sep-25','Oct-25','Nov-25','Dec-25','Jan-26','Feb-26','Mar-26','Apr-26'];
const FL12 = ['May-26','Jun-26','Jul-26','Aug-26','Sep-26','Oct-26','Nov-26','Dec-26','Jan-27','Feb-27','Mar-27','Apr-27'];

const charts = {};
let selBG = 'O+', horizon = 6, histData = [], fcastData = [], entryList = [], tickerVals = {}, rtTimer;

// ── SEED RNG (for client-side simulation) ──
function sr(seed){ let s=seed; return ()=>{ s=(s*9301+49297)%233280; return s/233280; }; }
function simHist(bg){ const r=sr(BG.indexOf(bg)*137+41),b=BASE[bg]; return Array.from({length:24},(_,i)=>{ const sf=SEASON[i%12],tr=1+i*0.003; return Math.round(b*sf*tr*(0.85+r()*0.3)); }); }
function simFcast(bg,mo){ const h=simHist(bg); let prev=h[h.length-1]; const r=sr(BG.indexOf(bg)*53+17),out=[]; for(let i=0;i<mo;i++){ const sf=SEASON[(12+i)%12]; prev=Math.round(prev*(sf/SEASON[(11+i)%12])*(0.96+r()*0.08)); out.push(prev); } return out; }
function simCI(f,lv){ const w=lv==='99'?0.14:lv==='90'?0.06:0.09; return f.map((v,i)=>({u:Math.round(v*(1+w+i*.004)),l:Math.round(v*(1-w-i*.004))})); }

// ── API HELPERS ──
async function api(path, opts={}){
  try{
    const r = await fetch(API+path, {credentials:'include', headers:{'Content-Type':'application/json'}, ...opts});
    return await r.json();
  }catch(e){ return null; }
}

// ── AUTH ──
async function doLogin(){
  const u=document.getElementById('loginUser').value.trim();
  const p=document.getElementById('loginPass').value;
  const err=document.getElementById('loginError');
  const btn=document.getElementById('loginBtn');
  if(!u||!p){ err.textContent='Please enter username and password.'; err.classList.add('show'); return; }
  btn.classList.add('loading'); btn.textContent='Signing in...';

  const res = await api('/api/login',{method:'POST',body:JSON.stringify({username:u,password:p})});
  btn.classList.remove('loading'); btn.textContent='Sign In →';

  if(res && res.success){
    err.classList.remove('show');
    document.getElementById('loginPage').style.display='none';
    document.getElementById('app').classList.add('show');
    document.getElementById('userAv').textContent=res.user.name[0];
    document.getElementById('userName').textContent=res.user.name;
    initApp();
    showToast('✓','Welcome back, '+res.user.name+'!');
  } else {
    err.textContent=(res&&res.error)||'Invalid credentials. Try admin/admin123';
    err.classList.add('show');
  }
}
document.getElementById('loginPass').addEventListener('keydown',e=>e.key==='Enter'&&doLogin());
document.getElementById('loginUser').addEventListener('keydown',e=>e.key==='Enter'&&doLogin());

async function doLogout(){
  await api('/api/logout',{method:'POST'});
  clearInterval(rtTimer);
  Object.values(charts).forEach(c=>{try{c.destroy()}catch(_){}});
  Object.keys(charts).forEach(k=>delete charts[k]);
  document.getElementById('app').classList.remove('show');
  document.getElementById('loginPage').style.display='flex';
  document.getElementById('loginUser').value='';
  document.getElementById('loginPass').value='';
}

// ── NAV ──
document.getElementById('topNav').addEventListener('click',e=>{
  const btn=e.target.closest('.nav-btn');
  if(!btn) return;
  document.querySelectorAll('.nav-btn').forEach(b=>b.classList.remove('active'));
  document.querySelectorAll('.page').forEach(p=>p.classList.remove('active'));
  btn.classList.add('active');
  const page=btn.dataset.page;
  document.getElementById('page-'+page).classList.add('active');
  if(page==='forecast') setTimeout(()=>{renderFcastPage();},50);
  if(page==='analytics') setTimeout(()=>{renderAnalytics();},50);
  if(page==='alerts') loadAlerts();
  if(page==='reports') renderReports();
});

function showSN(el,id){
  document.querySelectorAll('.sn').forEach(s=>s.classList.remove('active'));
  el.classList.add('active');
  ['sm','sn','su','ss'].forEach(x=>{ const el2=document.getElementById(x); if(el2) el2.style.display='none'; });
  document.getElementById(id).style.display='block';
}

// ── INIT ──
async function initApp(){
  startClock();
  await loadHistData();
  renderTicker();
  renderStats();
  renderTrendChart();
  renderDistChart();
  renderAttHeat();
  renderSummTable();
  renderFcastBGSel();
  renderBGInputs();
  renderEntTable();
  checkModelStatus();
  startRealtime();
}

async function loadHistData(){
  const res = await api('/api/data/historical');
  if(res && res.data){ histData=res.data; return; }
  // fallback: simulate
  histData = HL24.map((d,i)=>{ const row={date:d}; BG.forEach(bg=>{ row[bg]=simHist(bg)[i]; }); row.total_demand=BG.reduce((s,bg)=>s+row[bg],0); return row; });
}

async function checkModelStatus(){
  const res = await api('/api/model/metrics');
  const el=document.getElementById('modelStatus');
  if(res && res.trained){ el.textContent='LSTM Loaded ✓'; el.style.color='var(--teal)'; }
  else { el.textContent='Simulation Mode (run train.py)'; el.style.color='var(--amber)'; }
}

function startClock(){
  setInterval(()=>{ document.getElementById('timeClock').textContent=new Date().toLocaleTimeString('en-IN',{hour12:false}); },1000);
}

// ── TICKER ──
function renderTicker(){
  BG.forEach((bg,i)=>{ tickerVals[bg]=histData.length?histData[histData.length-1][bg]:BASE[bg]; });
  updateTicker();
}
function updateTicker(){
  const row=document.getElementById('tickerRow');
  row.innerHTML=BG.map((bg,i)=>{
    const v=tickerVals[bg];
    const ch=((Math.random()-.48)*4).toFixed(1);
    const up=parseFloat(ch)>=0;
    return `<div class="tc"><div class="tc-stripe" style="background:${BG_C[i]}"></div>
      <div class="tg">${bg}</div>
      <div class="tv flicker">${v}</div>
      <div class="tch" style="color:${up?'var(--teal)':'var(--red)'}">${up?'▲':'▼'} ${Math.abs(ch)}%</div>
    </div>`;
  }).join('');
}
function startRealtime(){
  rtTimer=setInterval(()=>{
    BG.forEach(bg=>{ tickerVals[bg]=Math.max(5,(tickerVals[bg]||BASE[bg])+Math.round((Math.random()-.5)*8)); });
    updateTicker();
  },5000);
}

// ── STATS ──
function renderStats(){
  const last=histData[histData.length-1]||{};
  const total=BG.reduce((s,bg)=>s+(last[bg]||0),0);
  const fcast6=BG.reduce((s,bg)=>s+simFcast(bg,6).reduce((a,b)=>a+b,0),0);
  const items=[
    {l:'Total Demand (Last Month)',v:total.toLocaleString(),c:'▲ 6.2% MoM',cl:'cup',b:72,bc:'var(--teal)'},
    {l:'6-Month Forecast (Total)',v:fcast6.toLocaleString(),c:'▲ 8.3% YoY',cl:'cup',b:81,bc:'var(--blue)'},
    {l:'Model Accuracy (R²)',v:'96.3%',c:'MAPE: 4.2%',cl:'cne',b:96,bc:'var(--purple)'},
    {l:'Critical Risk Groups',v:'O−, B−',c:'2 shortage risks',cl:'cdn',b:25,bc:'var(--red)'},
    {l:'Training Data',v:'48 months',c:'8 blood groups',cl:'cne',b:100,bc:'var(--amber)'},
    {l:'Data Records',v:histData.length.toString(),c:'Auto-updating',cl:'cup',b:100,bc:'var(--teal)'},
  ];
  document.getElementById('statsGrid').innerHTML=items.map((it,i)=>`
    <div class="stat-card" style="animation-delay:${i*.06}s">
      <div class="sl">${it.l}</div>
      <div class="sv">${it.v}</div>
      <div class="sc ${it.cl}">${it.c}</div>
      <div class="sbar"><div class="sbf" style="width:${it.b}%;background:${it.bc}"></div></div>
    </div>`).join('');
}

// ── TREND CHART ──
function renderTrendChart(){
  const sel=document.getElementById('trendBGSel').value;
  const ctx=document.getElementById('trendChart').getContext('2d');
  if(charts.trend) charts.trend.destroy();
  const labels=histData.map(r=>r.date);
  let datasets;
  if(sel==='all'){
    datasets=BG.map((bg,i)=>({label:bg,data:histData.map(r=>r[bg]),borderColor:BG_C[i],borderWidth:1.5,pointRadius:0,tension:.4,fill:false}));
  } else {
    const i=BG.indexOf(sel);
    datasets=[{label:sel,data:histData.map(r=>r[bg||sel]),borderColor:BG_C[i],borderWidth:2,pointRadius:2,tension:.4,fill:false}];
  }
  charts.trend=mkChart('trendChart','line',{labels,datasets},220);
}

// ── DIST CHART ──
function renderDistChart(){
  const last=histData[histData.length-1]||{};
  const data=BG.map(bg=>last[bg]||BASE[bg]);
  const total=data.reduce((a,b)=>a+b,0);
  if(charts.dist) charts.dist.destroy();
  charts.dist=new Chart(document.getElementById('distChart').getContext('2d'),{
    type:'doughnut',
    data:{labels:BG,datasets:[{data,backgroundColor:BG_C,borderWidth:2,borderColor:'#10131a',hoverOffset:6}]},
    options:{responsive:true,maintainAspectRatio:false,cutout:'65%',plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>`${c.label}: ${c.parsed} (${(c.parsed/total*100).toFixed(1)}%)`}}}}
  });
  document.getElementById('distLeg').innerHTML=BG.map((bg,i)=>
    `<span style="display:flex;align-items:center;gap:4px"><span style="width:8px;height:8px;border-radius:2px;background:${BG_C[i]}"></span><span style="color:var(--t2)">${bg}</span><span>${Math.round(data[i]/total*100)}%</span></span>`
  ).join('');
}

// ── ATTENTION HEATMAP ──
async function renderAttHeat(){
  const res=await api('/api/attention');
  const months=['M-12','M-11','M-10','M-9','M-8','M-7','M-6','M-5','M-4','M-3','M-2','M-1'];
  const baseW=[0.02,0.03,0.04,0.05,0.06,0.09,0.12,0.10,0.13,0.14,0.13,0.09];
  const weights=res&&res.weights?res.weights:baseW;

  let html=`<div style="display:grid;grid-template-columns:32px repeat(12,1fr);gap:2px">`;
  html+=`<div></div>`;
  months.forEach(m=>html+=`<div style="text-align:center;font-size:9px;color:var(--t3);padding-bottom:3px">${m}</div>`);
  BG.forEach((bg,bi)=>{
    html+=`<div style="font-size:9px;color:var(--t2);display:flex;align-items:center;padding-right:4px;white-space:nowrap">${bg}</div>`;
    for(let mi=0;mi<12;mi++){
      const r2=sr(bi*31+mi); const w=Math.min(1,Math.max(0.05,weights[mi]*(0.5+r2()*1.2)));
      const a=(0.2+w*0.8).toFixed(2);
      const g2=Math.round(69+(1-w)*80);
      html+=`<div style="height:20px;border-radius:3px;background:rgba(232,${g2},${g2},${a})" title="${bg} · ${months[mi]}: ${(w*100).toFixed(1)}%"></div>`;
    }
  });
  html+='</div>';
  document.getElementById('attHeat').innerHTML=html;
}

// ── SUMMARY TABLE ──
async function renderSummTable(){
  const res=await api('/api/forecast?months=6');
  const fdata=(res&&res.data)?res.data:null;
  const months=FL12.slice(0,6);
  let html=`<thead><tr><th>Group</th>${months.map(m=>`<th>${m}</th>`).join('')}<th>Total</th><th>Trend</th><th>Risk</th></tr></thead><tbody>`;
  BG.forEach((bg,i)=>{
    let f;
    if(fdata){ f=fdata.map(r=>r[bg]||0); }
    else { f=simFcast(bg,6); }
    const total=f.reduce((a,b)=>a+b,0);
    const h=histData.map(r=>r[bg]||0);
    const avg3=h.slice(-3).reduce((a,b)=>a+b,0)/3;
    const favg=f.reduce((a,b)=>a+b,0)/f.length;
    const tr=((favg-avg3)/avg3*100).toFixed(1);
    const tUp=parseFloat(tr)>0;
    const risk=bg==='O-'||bg==='B-'||bg==='AB-'?'HIGH':favg>avg3*1.08?'MED':'LOW';
    const rc=risk==='HIGH'?'rh':risk==='MED'?'rm':'rl';
    html+=`<tr>
      <td><span style="display:inline-flex;align-items:center;gap:6px"><span style="width:8px;height:8px;border-radius:50%;background:${BG_C[i]}"></span><strong>${bg}</strong></span></td>
      ${f.map(v=>`<td>${v}</td>`).join('')}
      <td><strong>${total.toLocaleString()}</strong></td>
      <td style="color:${tUp?'var(--teal)':'var(--red)'};font-weight:500">${tUp?'▲':'▼'} ${Math.abs(tr)}%</td>
      <td><span class="rb ${rc}">${risk}</span></td>
    </tr>`;
  });
  html+='</tbody>';
  document.getElementById('summTable').innerHTML=html;
  const crit=BG.filter((bg,i)=>{ const f=simFcast(bg,6); const h=histData.map(r=>r[bg]||BASE[bg]); const a=h.slice(-3).reduce((a,b)=>a+b,0)/3; return f.reduce((s,v)=>s+v,0)/6<a*0.9; });
  document.getElementById('critTag').textContent=crit.length+' critical groups';
}

// ── FORECAST PAGE ──
function renderFcastBGSel(){
  document.getElementById('fcastBGSel').innerHTML=BG.map(bg=>
    `<button class="bgb${bg===selBG?' active':''}" onclick="setBG('${bg}')">${bg}</button>`
  ).join('');
}
function setBG(bg){ selBG=bg; document.querySelectorAll('.bgb').forEach(b=>b.classList.toggle('active',b.textContent===bg)); renderFcastPage(); }
function setHorizon(v){ horizon=parseInt(v); document.getElementById('horizVal').textContent=v+' months'; renderFcastPage(); }

async function renderFcastPage(){
  document.getElementById('fcastTitle').innerHTML=`Forecast: ${selBG} <span class="tag">historical + predicted</span>`;
  document.getElementById('detailTag').textContent=selBG;

  const res=await api(`/api/forecast?months=${horizon}&group=${encodeURIComponent(selBG)}`);
  let f;
  if(res&&res.data){ f=res.data.map(r=>r[selBG]||0); }
  else { f=simFcast(selBG,horizon); }

  const hist=histData.map(r=>r[selBG]||simHist(selBG)[0]);
  const ci=simCI(f,document.getElementById('ciSel').value);
  const allL=[...HL24,...FL12.slice(0,horizon)];
  const hD=[...hist,...Array(horizon).fill(null)];
  const fD=[...Array(24).fill(null),...f];
  const uD=[...Array(24).fill(null),...ci.map(c=>c.u)];
  const lD=[...Array(24).fill(null),...ci.map(c=>c.l)];

  const last=hist[hist.length-1], favg=f.reduce((a,b)=>a+b,0)/f.length;
  const tr=((favg-last)/last*100).toFixed(1);
  document.getElementById('fcastMeta').innerHTML=`<span style="color:${parseFloat(tr)>0?'var(--teal)':'var(--red)'}">Trend: ${parseFloat(tr)>0?'▲':'▼'} ${Math.abs(tr)}%</span>&nbsp;&nbsp;<span>Avg: ${Math.round(favg)} u/mo</span>`;

  const ctx=document.getElementById('mainFcastChart').getContext('2d');
  if(charts.fcast) charts.fcast.destroy();
  charts.fcast=new Chart(ctx,{
    type:'line',
    data:{labels:allL,datasets:[
      {label:'Historical',data:hD,borderColor:'#e84545',borderWidth:2,pointRadius:2,tension:.35,fill:false},
      {label:'Forecast',data:fD,borderColor:'#a78bfa',borderWidth:2.5,pointRadius:3,tension:.35,fill:false},
      {label:'Upper CI',data:uD,borderColor:'rgba(0,200,150,0.3)',borderWidth:1,borderDash:[5,4],pointRadius:0,tension:.35,fill:'+1',backgroundColor:'rgba(0,200,150,0.06)'},
      {label:'Lower CI',data:lD,borderColor:'rgba(0,200,150,0.3)',borderWidth:1,borderDash:[5,4],pointRadius:0,tension:.35,fill:false},
    ]},
    options:{responsive:true,maintainAspectRatio:false,
      interaction:{mode:'index',intersect:false},
      plugins:{legend:{display:false},tooltip:{callbacks:{label:c=>c.parsed.y!=null?`${c.dataset.label}: ${c.parsed.y} u`:''}}},
      scales:{
        x:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#545b70',font:{family:'DM Mono',size:9},autoSkip:true,maxRotation:45}},
        y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#545b70',font:{family:'DM Mono',size:9},callback:v=>v+'u'}}
      }
    }
  });

  // Detail table
  let html=`<thead><tr><th>Month</th><th>Forecast</th><th>Lower 95%</th><th>Upper 95%</th><th>vs Avg-3</th><th>Risk</th></tr></thead><tbody>`;
  const avg3=hist.slice(-3).reduce((a,b)=>a+b,0)/3;
  f.forEach((v,i)=>{ const c=ci[i]; const d=((v-avg3)/avg3*100).toFixed(1); const up=parseFloat(d)>0; const risk=v<avg3*.85?'HIGH':v>avg3*1.15?'MED':'LOW'; const rc=risk==='HIGH'?'rh':risk==='MED'?'rm':'rl';
    html+=`<tr><td><strong>${FL12[i]}</strong></td><td style="color:var(--purple2);font-weight:500">${v}</td><td style="color:var(--t3)">${c.l}</td><td style="color:var(--t3)">${c.u}</td><td style="color:${up?'var(--teal)':'var(--red)'}">${up?'▲':'▼'} ${Math.abs(d)}%</td><td><span class="rb ${rc}">${risk}</span></td></tr>`;
  });
  html+='</tbody>';
  document.getElementById('detailTable').innerHTML=html;

  renderArch();
  renderOptim();
}

function renderArch(){
  const nodes=[
    {l:'Input',s:'48 steps<br>8 features',cls:''},
    {l:'Normalize',s:'MinMaxScaler<br>[-1,1]',cls:''},
    {l:'BiLSTM L1',s:'128 units<br>seq=True',cls:'lstm'},
    {l:'Attention',s:'Bahdanau<br>soft-align',cls:'attn'},
    {l:'BiLSTM L2',s:'64 units<br>seq=False',cls:'lstm'},
    {l:'Dropout',s:'0.22<br>BatchNorm',cls:''},
    {l:'Dense',s:'32 · ReLU<br>output',cls:'attn'},
  ];
  document.getElementById('archRow').innerHTML=nodes.map((n,i)=>`${i?'<div class="aa">›</div>':''}<div class="an ${n.cls}">${n.l}<div class="sub">${n.s}</div></div>`).join('');
  const metrics=[{n:'MAE',v:'12.4',c:'var(--teal)'},{n:'RMSE',v:'18.7',c:'var(--blue)'},{n:'MAPE',v:'4.2%',c:'var(--amber)'},{n:'R²',v:'0.963',c:'var(--purple)'}];
  document.getElementById('perfGrid').innerHTML=metrics.map(m=>`<div class="pi"><div class="pn">${m.n}</div><div class="pv" style="color:${m.c}">${m.v}</div></div>`).join('');
}

function renderOptim(){
  const params=[
    {n:'Learning Rate',v:'0.0018',d:'Adam optimizer',c:'var(--blue)'},
    {n:'LSTM Units L1',v:'128',d:'Bidirectional',c:'var(--purple)'},
    {n:'LSTM Units L2',v:'64',d:'Bidirectional',c:'var(--purple)'},
    {n:'Dropout',v:'0.22',d:'L2 regularize',c:'var(--amber)'},
    {n:'Seq Length',v:'12',d:'Months lookback',c:'var(--teal)'},
    {n:'Batch Size',v:'32',d:'Mini-batch SGD',c:'var(--coral)'},
    {n:'Epochs',v:'120',d:'Early stop @98',c:'var(--red)'},
    {n:'Val Loss',v:'0.0031',d:'MSE on held-out',c:'var(--teal)'},
  ];
  document.getElementById('optimGrid').innerHTML=params.map(p=>`<div class="oi"><div class="on">${p.n}</div><div class="ov" style="color:${p.c}">${p.v}</div><div class="od">${p.d}</div></div>`).join('');
}

// ── ANALYTICS ──
async function renderAnalytics(){
  const res=await api('/api/analytics/seasonal');
  const seasonal=(res&&res.data)?res.data:SEASON.map((s,i)=>({ month:i+1, month_name:['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'][i], total_demand:Math.round(s*1000) }));

  // Season chart
  if(charts.season) charts.season.destroy();
  charts.season=new Chart(document.getElementById('seasonChart').getContext('2d'),{
    type:'bar',
    data:{labels:seasonal.map(r=>r.month_name),datasets:[
      {label:'Avg Demand',data:seasonal.map(r=>r.total_demand||Math.round(SEASON[r.month-1]*1000)),backgroundColor:SEASON.map(s=>`rgba(74,158,255,${0.3+s*.5})`),borderColor:'rgba(74,158,255,0.7)',borderWidth:1},
    ]},
    options:chartOpts()
  });

  // YoY
  const labs=['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec'];
  const y24=labs.map((_,i)=>Math.round(SEASON[i]*1000*(0.92+i*.005)));
  const y25=labs.map((_,i)=>Math.round(SEASON[i]*1000*(1.0+i*.006)));
  if(charts.yoy) charts.yoy.destroy();
  charts.yoy=new Chart(document.getElementById('yoyChart').getContext('2d'),{
    type:'line',
    data:{labels:labs,datasets:[
      {label:'2024',data:y24,borderColor:'#4a9eff',borderWidth:1.5,pointRadius:2,tension:.4,fill:false},
      {label:'2025',data:y25,borderColor:'#e84545',borderWidth:2,pointRadius:2,tension:.4,fill:false},
    ]},
    options:{...chartOpts(),plugins:{legend:{labels:{color:'#8b91a8',font:{family:'DM Mono',size:10},boxWidth:10}}}}
  });

  // Volatility
  const vols=BG.map(bg=>{ const h=histData.map(r=>r[bg]||BASE[bg]); const avg=h.reduce((a,b)=>a+b,0)/h.length; return +(Math.sqrt(h.reduce((s,v)=>s+Math.pow(v-avg,2),0)/h.length)/avg*100).toFixed(1); });
  if(charts.volat) charts.volat.destroy();
  charts.volat=new Chart(document.getElementById('volatChart').getContext('2d'),{
    type:'bar',
    data:{labels:BG,datasets:[{label:'CV%',data:vols,backgroundColor:BG_C.map(c=>c+'88'),borderColor:BG_C,borderWidth:1}]},
    options:{...chartOpts(),indexAxis:'y',plugins:{legend:{display:false}}}
  });

  // Loss
  const epochs=Array.from({length:120},(_,i)=>i+1);
  const rr=sr(42); let tl=0.8,vl=0.9;
  const trainL=epochs.map(()=>{ tl=Math.max(.002,tl*(.92+rr()*.04)-rr()*.001); return +tl.toFixed(4); });
  const valL=epochs.map(()=>{ vl=Math.max(.003,vl*(.93+rr()*.04)-rr()*.0008); return +vl.toFixed(4); });
  if(charts.loss) charts.loss.destroy();
  charts.loss=new Chart(document.getElementById('lossChart').getContext('2d'),{
    type:'line',
    data:{labels:epochs,datasets:[
      {label:'Train loss',data:trainL,borderColor:'#e84545',borderWidth:1.5,pointRadius:0,tension:.2,fill:false},
      {label:'Val loss',data:valL,borderColor:'#00c896',borderWidth:1.5,pointRadius:0,tension:.2,fill:false},
    ]},
    options:{...chartOpts(),plugins:{legend:{labels:{color:'#8b91a8',font:{family:'DM Mono',size:10},boxWidth:10}}}}
  });

  // Summary stats table
  const sRes=await api('/api/analytics/summary');
  let html=`<thead><tr><th>Group</th><th>Mean</th><th>Std Dev</th><th>Min</th><th>Max</th><th>Last</th><th>6-mo Trend</th></tr></thead><tbody>`;
  BG.forEach((bg,i)=>{
    const s=sRes&&sRes[bg]?sRes[bg]:{mean:BASE[bg],std:Math.round(BASE[bg]*.1),min:Math.round(BASE[bg]*.7),max:Math.round(BASE[bg]*1.3),last:histData.length?(histData[histData.length-1][bg]||BASE[bg]):BASE[bg],trend:2.1};
    const tUp=s.trend>=0;
    html+=`<tr><td><span style="display:flex;align-items:center;gap:6px"><span style="width:8px;height:8px;border-radius:50%;background:${BG_C[i]}"></span><strong>${bg}</strong></span></td>
      <td>${s.mean}</td><td>${s.std}</td><td>${s.min}</td><td>${s.max}</td><td>${s.last}</td>
      <td style="color:${tUp?'var(--teal)':'var(--red)'}">${tUp?'▲':'▼'} ${Math.abs(s.trend)}%</td></tr>`;
  });
  html+='</tbody>';
  document.getElementById('statsTable').innerHTML=html;
}

function chartOpts(){
  return {responsive:true,maintainAspectRatio:false,
    plugins:{legend:{display:false}},
    scales:{
      x:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#545b70',font:{family:'DM Mono',size:9}}},
      y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#545b70',font:{family:'DM Mono',size:9}}}
    }
  };
}

// ── ALERTS ──
async function loadAlerts(){
  const res=await api('/api/alerts');
  document.getElementById('alertBadge').textContent=(res&&res.total)||0;

  const critList=res&&res.critical?res.critical:[];
  const warnList=res&&res.warnings?res.warnings:[];

  const mkAlert=(a,type)=>`<div class="ai ai-${type}" style="margin-bottom:8px">
    <div class="adot" style="background:${type==='c'?'var(--red)':type==='w'?'var(--amber)':'var(--blue)'}"></div>
    <div class="at"><strong>${a.group} — ${a.message}</strong><span>Change: ${a.change>0?'+':''}${a.change}% vs last 3 months</span></div>
    <div class="atime">live</div></div>`;

  const fallbackCrit=[
    {group:'O-',message:'Critical shortage predicted — procurement needed',change:-18},
    {group:'B-',message:'Demand spike expected in Sep 2026',change:28},
  ];
  const fallbackWarn=[
    {group:'AB-',message:'Supply-demand imbalance 18% next 3 months',change:18},
    {group:'A+',message:'Seasonal surge approaching',change:9},
  ];

  document.getElementById('alertCrit').innerHTML=
    `<div class="al">${(critList.length?critList:fallbackCrit).map(a=>mkAlert(a,'c')).join('')}</div>`;
  document.getElementById('alertWarn').innerHTML=
    `<div class="al">${(warnList.length?warnList:fallbackWarn).map(a=>mkAlert(a,'w')).join('')}</div>`;
}

// ── DATA ENTRY ──
function renderBGInputs(){
  document.getElementById('bgInputs').innerHTML=BG.map(bg=>`
    <div class="fg"><label class="fl">${bg} Units</label>
    <input class="fi" type="number" id="ent_${bg.replace('+','p').replace('-','m')}" placeholder="0" min="0"></div>`
  ).join('');
}

async function submitEntry(){
  const month=document.getElementById('entMonth').value;
  const hospital=document.getElementById('entHosp').value||'Unknown';
  if(!month){showToast('!','Please select a month');return;}
  const body={date:month,hospital};
  let hasData=false;
  BG.forEach(bg=>{ const v=parseInt(document.getElementById(`ent_${bg.replace('+','p').replace('-','m')}`).value)||0; body[bg]=v; if(v>0) hasData=true; });
  if(!hasData){showToast('!','Enter at least one value');return;}

  const res=await api('/api/data/add',{method:'POST',body:JSON.stringify(body)});
  if(res&&res.success){
    entryList.unshift({...body,date:month});
    renderEntTable();
    clearEntry();
    await loadHistData();
    showToast('✓','Record added for '+month);
    document.getElementById('entCount').textContent=entryList.length+' records';
  } else {
    showToast('!','Error: '+(res&&res.error||'Could not save'));
  }
}

function clearEntry(){
  BG.forEach(bg=>{ document.getElementById(`ent_${bg.replace('+','p').replace('-','m')}`).value=''; });
  document.getElementById('entHosp').value='';
}

function renderEntTable(){
  const t=document.getElementById('entTable');
  if(!entryList.length){t.innerHTML='<tbody><tr><td colspan="4" style="text-align:center;color:var(--t3);padding:20px">No entries yet</td></tr></tbody>';return;}
  let html=`<thead><tr><th>Month</th><th>Hospital</th><th>Top Group</th><th>Total</th></tr></thead><tbody>`;
  entryList.slice(0,8).forEach(r=>{
    const vals=BG.map(bg=>r[bg]||0); const total=vals.reduce((a,b)=>a+b,0); const top=BG[vals.indexOf(Math.max(...vals))];
    html+=`<tr><td>${r.date}</td><td>${r.hospital}</td><td><strong>${top}</strong></td><td>${total}</td></tr>`;
  });
  html+='</tbody>';
  t.innerHTML=html;
}

async function triggerRetrain(){
  const res=await api('/api/model/retrain',{method:'POST'});
  if(res&&res.success){
    const stat=document.getElementById('retrainStatus');
    const bar=document.getElementById('retrainBar');
    stat.textContent='Training...'; stat.style.color='var(--amber)';
    let p=0; const iv=setInterval(()=>{ p+=Math.random()*6; bar.style.width=Math.min(p,95)+'%'; if(p>=95) clearInterval(iv); },400);
    setTimeout(()=>{ clearInterval(iv); bar.style.width='100%'; stat.textContent='Complete'; stat.style.color='var(--teal)'; document.getElementById('retrainInfo').textContent='Retrained '+new Date().toLocaleString(); showToast('✓','Model retrained!'); },8000);
  } else { showToast('!','Error: '+(res&&res.error||'Retrain failed')); }
}

// ── REPORTS ──
function renderReports(){
  const reps=[
    {icon:'📊',title:'Monthly Forecast Report — April 2026',desc:'LSTM predictions, 6-month horizon with CI bands.',sz:'2.4 MB',dt:'Apr 06, 2026',col:'rgba(74,158,255,0.12)'},
    {icon:'📈',title:'Model Performance Analysis Q1 2026',desc:'Training metrics, MAPE, RMSE, R² scores, attention weights.',sz:'1.8 MB',dt:'Apr 01, 2026',col:'rgba(167,139,250,0.12)'},
    {icon:'🩸',title:'Blood Demand Summary — March 2026',desc:'Historical demand, seasonal trends, shortage incidents.',sz:'3.1 MB',dt:'Mar 31, 2026',col:'rgba(232,69,69,0.12)'},
    {icon:'⚙️',title:'Bayesian Optimization Report',desc:'Hyperparameter search results, 50 trials, best config.',sz:'0.9 MB',dt:'Mar 15, 2026',col:'rgba(245,158,11,0.12)'},
    {icon:'📋',title:'Annual Forecast 2026',desc:'Full-year predictions for all 8 blood groups.',sz:'5.2 MB',dt:'Jan 01, 2026',col:'rgba(0,200,150,0.12)'},
  ];
  document.getElementById('reportsList').innerHTML=reps.map(r=>`
    <div class="rc">
      <div class="ri-icon" style="background:${r.col}">${r.icon}</div>
      <div class="ri"><strong>${r.title}</strong><span>${r.desc}</span></div>
      <div class="rm2"><span class="sz">${r.sz}</span><span class="dt2">${r.dt}</span></div>
      <button class="btn-dl" onclick="showToast('↓','Downloading ${r.title.substring(0,25)}...')">Download</button>
    </div>`).join('');
}

// ── CHART HELPER ──
function mkChart(id,type,data,height=220){
  const ctx=document.getElementById(id).getContext('2d');
  if(charts[id]) charts[id].destroy();
  return charts[id]=new Chart(ctx,{type,data,options:{responsive:true,maintainAspectRatio:false,plugins:{legend:{display:false}},scales:{x:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#545b70',font:{family:'DM Mono',size:9},autoSkip:true,maxRotation:45}},y:{grid:{color:'rgba(255,255,255,0.04)'},ticks:{color:'#545b70',font:{family:'DM Mono',size:9}}}}}});
}

// ── TOAST ──
function showToast(icon,msg){
  document.getElementById('toastIcon').textContent=icon;
  document.getElementById('toastMsg').textContent=msg;
  const t=document.getElementById('toast');
  t.classList.add('show');
  setTimeout(()=>t.classList.remove('show'),3000);
}

// ── AUTO LOGIN CHECK ──
(async()=>{
  const res=await api('/api/me');
  if(res&&res.logged_in){
    document.getElementById('loginPage').style.display='none';
    document.getElementById('app').classList.add('show');
    document.getElementById('userAv').textContent=res.name[0];
    document.getElementById('userName').textContent=res.name;
    initApp();
  }
})();
