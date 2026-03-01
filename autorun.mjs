import CDP from "chrome-remote-interface";

const HOST = "127.0.0.1";
const PORT = 9222;

const CLICKER = `
(() => {
  if (window.__agAutorunInstalled) return "already";
  window.__agAutorunInstalled = true;

  const seen = new WeakSet();
  const norm = s => (s || "").replace(/\\s+/g," ").trim();

  function isRun(btn){
    const t = norm(btn.innerText || "");
    return /Run/i.test(t) && /Alt/i.test(t);
  }

  function scan(){
    let count = 0;
    document.querySelectorAll("button").forEach(btn=>{
      if(!isRun(btn)) return;
      if(btn.disabled) return;
      if(seen.has(btn)) return;
      seen.add(btn);
      btn.click();
      count++;
    });
    return count;
  }

  new MutationObserver(scan)
    .observe(document, {subtree:true,childList:true});

  setInterval(scan,200);
  return "installed";
})()
`;

const attached = new Set();

async function attach(target){
  if(attached.has(target.id)) return;

  try{
    const client = await CDP({target, host:HOST, port:PORT});
    attached.add(target.id);

    const {Runtime} = client;
    await Runtime.enable();

    const result = await Runtime.evaluate({
      expression: CLICKER,
      returnByValue:true
    });

    console.log("Injected into:", target.title || target.url, "=>", result.result.value);
  }
  catch(e){
    console.log("Attach failed:", e.message);
  }
}

async function loop(){
  try{
    const targets = await CDP.List({host:HOST, port:PORT});
    console.log("Targets:", targets.length);

    for(const t of targets){
      if(t.type === "page" || t.type === "iframe" || t.type === "other"){
        await attach(t);
      }
    }
  }
  catch(e){
    console.log("CDP error:", e.message);
  }
}

setInterval(loop,1000);
console.log("Autorun bridge active...");