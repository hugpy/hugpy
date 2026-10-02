var __dsPreview = (() => {
  var __create = Object.create;
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __getProtoOf = Object.getPrototypeOf;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __esm = (fn, res, err) => function __init() {
    if (err) throw err[0];
    try {
      return fn && (res = (0, fn[__getOwnPropNames(fn)[0]])(fn = 0)), res;
    } catch (e) {
      throw err = [e], e;
    }
  };
  var __commonJS = (cb, mod) => function __require() {
    try {
      return mod || (0, cb[__getOwnPropNames(cb)[0]])((mod = { exports: {} }).exports, mod), mod.exports;
    } catch (e) {
      throw mod = 0, e;
    }
  };
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __reExport = (target, mod, secondTarget) => (__copyProps(target, mod, "default"), secondTarget && __copyProps(secondTarget, mod, "default"));
  var __toESM = (mod, isNodeMode, target) => (target = mod != null ? __create(__getProtoOf(mod)) : {}, __copyProps(
    // If the importer is in node compatibility mode or this is not an ESM
    // file that has been converted to a CommonJS file using a Babel-
    // compatible transform (i.e. "__esModule" has not been set), then set
    // "default" to the CommonJS "module.exports" for node compatibility.
    isNodeMode || !mod || !mod.__esModule ? __defProp(target, "default", { value: mod, enumerable: true }) : target,
    mod
  ));
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // <define:import.meta.env>
  var init_define_import_meta_env = __esm({
    "<define:import.meta.env>"() {
    }
  });

  // ds-raw:__ds_raw__
  var require_ds_raw = __commonJS({
    "ds-raw:__ds_raw__"(exports, module) {
      init_define_import_meta_env();
      module.exports = window.HugpyUI;
    }
  });

  // shim:react-shim
  var require_react_shim = __commonJS({
    "shim:react-shim"(exports, module) {
      init_define_import_meta_env();
      var R = window.React;
      function jsx2(t, p, k) {
        return R.createElement(t, k === void 0 ? p : Object.assign({ key: k }, p));
      }
      module.exports = R;
      module.exports.jsx = jsx2;
      module.exports.jsxs = jsx2;
      module.exports.jsxDEV = jsx2;
      module.exports.Fragment = R.Fragment;
    }
  });

  // .design-sync/previews/HugpyConsole.tsx
  var HugpyConsole_exports = {};
  __export(HugpyConsole_exports, {
    Console: () => Console
  });
  init_define_import_meta_env();

  // ds-shim:ds
  var ds_exports = {};
  __export(ds_exports, {
    default: () => ds_default
  });
  init_define_import_meta_env();
  __reExport(ds_exports, __toESM(require_ds_raw()));
  var g = window.HugpyUI;
  var ds_default = "default" in g ? g.default : g;

  // src/showroom/demoFetch.js
  init_define_import_meta_env();

  // src/showroom/fixtures.js
  init_define_import_meta_env();
  var J = (s) => JSON.parse(s);
  var AUTH_CONFIG = { mode: "open", base: null };
  var VERSION = { name: "hugpy", version: "0.1.47", api: 1, auth_mode: "open" };
  var READINESS = J(String.raw`{"central":{"version":"0.1.47","auth_mode":"open"},"storage":{"root":"/srv/hugpy/models","exists":true,"writable":true,"free_gb":312.6,"model_count":5},"serving":{"enabled":true,"any_serving":true,"slots":[{"slot":0,"model_key":"Qwen2.5-3B-Instruct-GGUF","healthy":true,"loaded":true}]},"console":{"configured":false,"url":null},"connect":{"base_url":"http://127.0.0.1:7002"}}`);
  var AUTH_ME = { authenticated: false };
  var MODELS = J(String.raw`[{"name":"Qwen2.5-3B-Instruct-GGUF","model_key":"Qwen2.5-3B-Instruct-GGUF","hub_id":"Qwen/Qwen2.5-3B-Instruct-GGUF","folder":"Qwen/Qwen2.5-3B-Instruct-GGUF","framework":"llama_cpp","tasks":["text-generation"],"primary_task":"text-generation","base_model":null,"model_max_length":32768,"filename":"qwen2.5-3b-instruct-q4_k_m.gguf","include":null,"port":null,"host":null,"timeout_s":3600,"meta":null,"extra":{},"pipeline_tag":"text-generation","parameter_count":3090000000,"license":"apache-2.0","languages":["en"],"tags":["text-generation","gguf","qwen2","chat"],"total_bytes":2018000000,"status":"installed","destination":"/srv/hugpy/models/gguf/text-generation/Qwen/Qwen2.5-3B-Instruct-GGUF","installed_marker":"/srv/hugpy/models/gguf/text-generation/Qwen/Qwen2.5-3B-Instruct-GGUF/hugpy.json","media":true},{"name":"Qwen2.5-VL-3B-Instruct-GGUF","model_key":"Qwen2.5-VL-3B-Instruct-GGUF","hub_id":"ggml-org/Qwen2.5-VL-3B-Instruct-GGUF","folder":"ggml-org/Qwen2.5-VL-3B-Instruct-GGUF","framework":"llama_cpp","tasks":["image-text-to-text","text-generation"],"primary_task":"image-text-to-text","base_model":null,"model_max_length":32768,"filename":"Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf","include":["Qwen2.5-VL-3B-Instruct-Q4_K_M.gguf","mmproj-Qwen2.5-VL-3B-Instruct-f16.gguf"],"port":null,"host":null,"timeout_s":3600,"meta":null,"extra":{},"pipeline_tag":"image-text-to-text","parameter_count":3750000000,"license":"apache-2.0","languages":["en"],"tags":["image-text-to-text","multimodal","vision","gguf"],"total_bytes":3650000000,"status":"installed","destination":"/srv/hugpy/models/gguf/image-text-to-text/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF","installed_marker":"/srv/hugpy/models/gguf/image-text-to-text/ggml-org/Qwen2.5-VL-3B-Instruct-GGUF/hugpy.json","media":true},{"name":"whisper-large-v3-turbo","model_key":"whisper-large-v3-turbo","hub_id":"openai/whisper-large-v3-turbo","folder":"openai/whisper-large-v3-turbo","framework":"transformers","tasks":["automatic-speech-recognition"],"primary_task":"automatic-speech-recognition","base_model":null,"model_max_length":448,"filename":null,"include":null,"port":null,"host":null,"timeout_s":3600,"meta":null,"extra":{},"pipeline_tag":"automatic-speech-recognition","parameter_count":809000000,"license":"mit","languages":["en","es","fr","de","zh"],"tags":["automatic-speech-recognition","whisper","audio"],"total_bytes":1620000000,"status":"installed","destination":"/srv/hugpy/models/transformers/automatic-speech-recognition/openai/whisper-large-v3-turbo","installed_marker":"/srv/hugpy/models/transformers/automatic-speech-recognition/openai/whisper-large-v3-turbo/hugpy.json","media":false},{"name":"all-minilm-l6-v2","model_key":"all-minilm-l6-v2","hub_id":"sentence-transformers/all-minilm-l6-v2","folder":"sentence-transformers/all-minilm-l6-v2","framework":"transformers","tasks":["feature-extraction","sentence-similarity","keyword-extraction"],"primary_task":"feature-extraction","base_model":null,"model_max_length":512,"filename":null,"include":null,"port":null,"host":null,"timeout_s":3600,"meta":null,"extra":{},"pipeline_tag":"feature-extraction","parameter_count":22700000,"license":"apache-2.0","languages":["en"],"tags":["feature-extraction","sentence-transformers","embeddings"],"total_bytes":91000000,"status":"installed","destination":"/srv/hugpy/models/transformers/feature-extraction/sentence-transformers/all-minilm-l6-v2","installed_marker":"/srv/hugpy/models/transformers/feature-extraction/sentence-transformers/all-minilm-l6-v2/hugpy.json","media":false},{"name":"flan-t5-large","model_key":"flan-t5-large","hub_id":"google/flan-t5-large","folder":"google/flan-t5-large","framework":"transformers","tasks":["text-summarization","text2text-generation"],"primary_task":"text-summarization","base_model":null,"model_max_length":1024,"filename":null,"include":null,"port":null,"host":null,"timeout_s":3600,"meta":null,"extra":{},"pipeline_tag":"text2text-generation","parameter_count":783000000,"license":"apache-2.0","languages":["en","fr","de"],"tags":["text2text-generation","summarization","t5"],"total_bytes":3050000000,"status":"installed","destination":"/srv/hugpy/models/transformers/text-summarization/google/flan-t5-large","installed_marker":"/srv/hugpy/models/transformers/text-summarization/google/flan-t5-large/hugpy.json","media":false},{"name":"sd-turbo","model_key":"sd-turbo","hub_id":"stabilityai/sd-turbo","folder":"stabilityai/sd-turbo","framework":"transformers","tasks":["text-to-image"],"primary_task":"text-to-image","base_model":null,"model_max_length":77,"filename":null,"include":null,"port":null,"host":null,"timeout_s":3600,"meta":null,"extra":{},"pipeline_tag":"text-to-image","parameter_count":865000000,"license":"other","languages":["en"],"tags":["text-to-image","diffusers","stable-diffusion"],"total_bytes":2580000000,"status":"partial","destination":"/srv/hugpy/models/transformers/text-to-image/stabilityai/sd-turbo","installed_marker":"/srv/hugpy/models/transformers/text-to-image/stabilityai/sd-turbo/hugpy.json","media":false},{"name":"Qwen2.5-VL-7B-Instruct-GGUF","model_key":"Qwen2.5-VL-7B-Instruct-GGUF","hub_id":"ggml-org/Qwen2.5-VL-7B-Instruct-GGUF","folder":"gguf/image-text-to-text/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF","framework":"llama_cpp","tasks":["image-text-to-text","text-generation"],"primary_task":"image-text-to-text","base_model":null,"model_max_length":32768,"filename":"Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf","include":["Qwen2.5-VL-7B-Instruct-Q4_K_M.gguf","mmproj-Qwen2.5-VL-7B-Instruct-f16.gguf"],"port":null,"host":null,"timeout_s":3600,"meta":{"hub_id":"ggml-org/Qwen2.5-VL-7B-Instruct-GGUF","pipeline_tag":"image-text-to-text","library_name":"llama.cpp","parameter_count":8290000000,"license":"apache-2.0","languages":["en"],"tags":["image-text-to-text","multimodal","vision","gguf"],"gated":false},"extra":{"dir":null,"serveable":true,"unserveable_tasks":[]},"pipeline_tag":"image-text-to-text","parameter_count":8290000000,"license":"apache-2.0","languages":["en"],"tags":["image-text-to-text","multimodal","vision","gguf"],"total_bytes":5500000000,"status":"not_installed","destination":"/srv/hugpy/models/gguf/image-text-to-text/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF","installed_marker":"/srv/hugpy/models/gguf/image-text-to-text/ggml-org/Qwen2.5-VL-7B-Instruct-GGUF/hugpy.json","media":false}]`);
  var MODEL_BY_KEY = J(String.raw`{"key":"Qwen2.5-3B-Instruct-GGUF","name":"Qwen2.5-3B-Instruct-GGUF","model_key":"Qwen2.5-3B-Instruct-GGUF","hub_id":"Qwen/Qwen2.5-3B-Instruct-GGUF","folder":"Qwen/Qwen2.5-3B-Instruct-GGUF","framework":"llama_cpp","tasks":["text-generation"],"primary_task":"text-generation","base_model":null,"model_max_length":32768,"filename":"qwen2.5-3b-instruct-q4_k_m.gguf","include":null,"port":null,"host":null,"timeout_s":3600,"meta":null,"extra":{},"status":"installed","destination":"/srv/hugpy/models/gguf/text-generation/Qwen/Qwen2.5-3B-Instruct-GGUF","installed_marker":"/srv/hugpy/models/gguf/text-generation/Qwen/Qwen2.5-3B-Instruct-GGUF/hugpy.json"}`);
  var V1_MODELS = J(String.raw`{"object":"list","data":[{"id":"Qwen2.5-3B-Instruct-GGUF","object":"model","created":0,"owned_by":"hugpy","hub_id":"Qwen/Qwen2.5-3B-Instruct-GGUF","task":"text-generation","context_length":32768},{"id":"Qwen2.5-VL-3B-Instruct-GGUF","object":"model","created":0,"owned_by":"hugpy","hub_id":"ggml-org/Qwen2.5-VL-3B-Instruct-GGUF","task":"image-text-to-text","context_length":32768},{"id":"whisper-large-v3-turbo","object":"model","created":0,"owned_by":"hugpy","hub_id":"openai/whisper-large-v3-turbo","task":"automatic-speech-recognition","context_length":448},{"id":"all-minilm-l6-v2","object":"model","created":0,"owned_by":"hugpy","hub_id":"sentence-transformers/all-minilm-l6-v2","task":"feature-extraction","context_length":512},{"id":"flan-t5-large","object":"model","created":0,"owned_by":"hugpy","hub_id":"google/flan-t5-large","task":"text-summarization","context_length":1024}]}`);
  var ML = J(String.raw`{"endpoints":{"/ml/transcribe":{"task":"automatic-speech-recognition","ready":true,"extra":"audio"},"/ml/summarize":{"task":"text-summarization","ready":true,"extra":"transformers"},"/ml/keywords":{"task":"keyword-extraction","ready":false,"extra":"keywords"},"/ml/embed":{"task":"feature-extraction","ready":true,"extra":"embed"},"/ml/similarity":{"task":"sentence-similarity","ready":true,"extra":"embed"},"/ml/vision":{"task":"image-text-to-text","ready":true,"extra":"engine"},"/ml/imagine":{"task":"text-to-image","ready":false,"extra":"imagegen"},"/ml/extract":{"task":"document-extraction","ready":true,"extra":"extract"},"/ml/fetch":{"task":"url-extraction","ready":true,"extra":"web"}},"pool":"ml"}`);
  var ML_GATE = { require_key: false };
  var PROMPT_TASKS = J(String.raw`{"tasks":["text-generation","image-text-to-text","automatic-speech-recognition","text-summarization","text2text-generation","feature-extraction","sentence-similarity","text-to-image","keyword-extraction"],"defaults":{"text-generation":"Qwen2.5-3B-Instruct-GGUF","image-text-to-text":"Qwen2.5-VL-3B-Instruct-GGUF","automatic-speech-recognition":"whisper-large-v3-turbo","text-summarization":"flan-t5-large","text2text-generation":"flan-t5-large","feature-extraction":"all-minilm-l6-v2","sentence-similarity":"all-minilm-l6-v2","text-to-image":"sd-turbo","keyword-extraction":"all-minilm-l6-v2"}}`);
  var KEYS = J(String.raw`{"require_key":false,"keys":[{"id":"a1b2c3d4e5f60718","name":"laptop","pool":"","prefix":"hp_3f9a2b1c","created_at":1750982400.0,"last_used":1751049600.0,"revoked":false},{"id":"99aa88bb77cc66dd","name":"cron","pool":"ml","prefix":"hp_7e1d4c8a","created_at":1750809600.0,"last_used":null,"revoked":false}]}`);
  var JOB = J(String.raw`{"id":"3f8c1d2a-9b4e-4c7a-8e1f-2a6b5c0d9e11","model_key":"sd-turbo","status":"running","message":"Downloading…","error":null,"progress":0.5183,"total_bytes":2580000000,"downloaded_bytes":1337000000,"attempt":1,"max_attempts":4,"stalled":false,"bytes_per_second":14300000.0,"created_at":"2026-06-27T17:42:11.103284+00:00","updated_at":"2026-06-27T17:43:42.880113+00:00"}`);
  var QUEUE = J(String.raw`{"active":[{"request_id":"req-1b8e44da","model_key":"Qwen2.5-3B-Instruct-GGUF","model":"Qwen2.5-3B-Instruct-GGUF","kind":"chat","state":"waiting","elapsed":1.1,"wait":1.1,"tokens":0},{"request_id":"req-7f3a91c2","model_key":"Qwen2.5-3B-Instruct-GGUF","model":"Qwen2.5-3B-Instruct-GGUF","kind":"chat","state":"active","elapsed":4.2,"wait":0.3,"tokens":142}],"counts":{"waiting":1,"active":1,"total":2}}`);
  var SLOTS = J(String.raw`{"enabled":true,"slots":[{"_control":"http://127.0.0.1:8101","slot_id":"1","control_port":8101,"child_port":9101,"endpoint":"http://127.0.0.1:8101","model_key":"Qwen2.5-3B-Instruct-GGUF","healthy":true,"n_gpu_layers":0,"ctx":8192,"threads":6,"cpus":null,"gpu":null,"allowed_cpus":null,"loaded_at":1782556800.0,"last_used":1782559740.0,"free_vram_bytes":null,"rss_bytes":3456106496,"expected_bytes":2178633344},{"_control":"http://127.0.0.1:8102","slot_id":"2","control_port":8102,"child_port":9102,"endpoint":"http://127.0.0.1:8102","model_key":null,"healthy":false,"n_gpu_layers":null,"ctx":null,"threads":null,"cpus":null,"gpu":null,"allowed_cpus":null,"loaded_at":0.0,"last_used":0.0,"free_vram_bytes":null,"rss_bytes":0,"expected_bytes":null}],"resources":{"total_bytes":67437129728,"available_bytes":49019215872,"free_bytes":9663676416,"cache_bytes":40802189312,"used_bytes":16971264000,"cpu_count":12}}`);
  var CACHE = J(String.raw`{"enabled":true,"dir":"/var/cache/hugpy-models","max_bytes":483183820800,"used_bytes":16087536768,"free_bytes":441879171072,"warming":["/srv/hugpy/models/Qwen2.5-Coder-7B-Instruct/qwen2.5-coder-7b-instruct-q4_k_m.gguf"],"entries":[{"dir":"Qwen2.5-3B-Instruct-GGUF","bytes":2178633344,"mtime":1782556800},{"dir":"Meta-Llama-3.1-8B-Instruct","bytes":4920536064,"mtime":1782490000},{"dir":"Qwen2.5-14B-Instruct","bytes":8988367360,"mtime":1782300000}]}`);
  var SERVING = J(String.raw`[{"key":"Qwen2.5-3B-Instruct-GGUF","mode":"swap","always_on":false,"endpoint":"http://127.0.0.1:9292","model_name":"qwen2.5-3b-instruct-gguf","port":9292,"n_gpu_layers":0,"threads":6,"ctx_size":8192,"ttl_seconds":600,"override":{}},{"key":"nomic-embed-text-v1.5","mode":"systemd","always_on":true,"endpoint":"http://127.0.0.1:8731","model_name":"nomic-embed-text-v1.5","port":8731,"n_gpu_layers":0,"threads":4,"ctx_size":2048,"ttl_seconds":null,"override":{"serve_mode":"systemd"},"unit":{"unit":"llama-nomic-embed-text-v1.5.service","state":"active","sub_state":"running","active":true,"rss_bytes":612368384,"since":1782470400}},{"key":"Meta-Llama-3.1-8B-Instruct","mode":"systemd","always_on":true,"endpoint":"http://127.0.0.1:8721","model_name":"meta-llama-3.1-8b-instruct","port":8721,"n_gpu_layers":0,"threads":8,"ctx_size":8192,"ttl_seconds":null,"override":{}},{"key":"Qwen2.5-14B-Instruct","mode":"off","always_on":false,"endpoint":null,"model_name":"qwen2.5-14b-instruct","port":8722,"n_gpu_layers":0,"threads":8,"ctx_size":8192,"ttl_seconds":600,"override":{"serve_mode":"off"}}]`);
  var SERVING_BY_KEY = J(String.raw`{"key":"Qwen2.5-3B-Instruct-GGUF","mode":"swap","always_on":false,"endpoint":"http://127.0.0.1:8081/v1","model_name":"Qwen2.5-3B-Instruct-GGUF","port":8081,"n_gpu_layers":-1,"threads":8,"ctx_size":8192,"ttl_seconds":900,"override":{"serve_mode":"swap","n_gpu_layers":-1},"available_gguf":["qwen2.5-3b-instruct-q4_k_m.gguf","qwen2.5-3b-instruct-q5_k_m.gguf"],"gguf_file":"qwen2.5-3b-instruct-q4_k_m.gguf"}`);
  var SLOTS_INSTALL = { steps: [] };
  var WORKERS = J(String.raw`[{"id":"a1f3c9d2e8b74f60","name":"gpu-box-1","url":"http://10.8.0.5:9100","role":"worker","gpus":[{"index":0,"name":"NVIDIA GeForce RTX 3090","memory_total":25769803776,"memory_free":1610612736}],"models":["Llama-3.1-8B-Instruct-GGUF","Qwen2.5-Coder-7B-Instruct-GGUF"],"pkg_version":"0.1.402","rpc_endpoint":null,"free_ram":33285996544,"engine":{"installed":true,"version":"0.3.2","supports_gpu_offload":true},"pool":"","admission":"approved","created_at":1782001234.51,"last_seen":1782518390.22,"last_picked":1782518301.0,"loaded_models":["Qwen2.5-Coder-7B-Instruct-GGUF"],"provisioning":["Llama-3.1-8B-Instruct-GGUF"],"provision_progress":{"Llama-3.1-8B-Instruct-GGUF":{"done_bytes":3221225472,"total_bytes":4831838208,"frac":0.667}},"spill":{"mode":"auto","n_gpu_layers_env":null,"gpu_mem_gib":null,"cpu_mem_gib":null,"tensor_split":null,"free_vram_bytes":1610612736},"spill_by_model":{"Llama-3.1-8B-Instruct-GGUF":{"gpu_mem_gib":20}},"status":"online"},{"id":"b7e2d4a1c6f09e35","name":"gpu-box-2","url":"http://10.8.0.6:9100","role":"worker","gpus":[{"index":0,"name":"NVIDIA GeForce RTX 4090","memory_total":25769803776,"memory_free":23622320128}],"models":["Qwen2.5-VL-7B-Instruct"],"pkg_version":"0.1.402","rpc_endpoint":null,"free_ram":50457796608,"engine":{"installed":true,"version":"0.3.2","supports_gpu_offload":true},"pool":"media","admission":"approved","created_at":1782010100.0,"last_seen":1782518388.91,"last_picked":1782517000.0,"loaded_models":[],"provisioning":[],"provision_progress":{},"spill":{"mode":"auto","n_gpu_layers_env":null,"gpu_mem_gib":null,"cpu_mem_gib":null,"tensor_split":null,"free_vram_bytes":23622320128},"spill_by_model":{},"status":"online"},{"id":"c3a8f1b9d2e74c08","name":"cpu-box-1","url":"http://10.8.0.9:9100","role":"worker","gpus":[],"models":[],"pkg_version":"0.1.402","rpc_endpoint":null,"free_ram":16579448832,"engine":{"installed":true,"version":"0.3.2","supports_gpu_offload":false},"pool":"","admission":"pending","created_at":1782518100.0,"last_seen":1782518389.5,"loaded_models":[],"provisioning":[],"spill":{"mode":"auto","n_gpu_layers_env":null,"gpu_mem_gib":null,"cpu_mem_gib":null,"tensor_split":null,"free_vram_bytes":null},"spill_by_model":{},"status":"online"}]`);
  var WORKER_HEALTH = J(String.raw`{"reachable":true,"url":"http://10.8.0.5:9100/health","health":{"ok":true,"worker_id":"a1f3c9d2e8b74f60","name":"gpu-box-1","gpus":[{"index":0,"name":"NVIDIA GeForce RTX 3090","memory_total":25769803776,"memory_free":1543503872}],"cuda":{"available":true,"device_count":1,"device_name":"NVIDIA GeForce RTX 3090","torch_version":"2.4.1+cu124","cuda_version":"12.4"},"llama_cpp":{"installed":true,"version":"0.3.2","supports_gpu_offload":true},"assigned_models":["Llama-3.1-8B-Instruct-GGUF","Qwen2.5-Coder-7B-Instruct-GGUF"],"provisioning":["Llama-3.1-8B-Instruct-GGUF"],"provision_progress":{"Llama-3.1-8B-Instruct-GGUF":{"done_bytes":3221225472,"total_bytes":4831838208,"frac":0.667}},"loaded_models":["Qwen2.5-Coder-7B-Instruct-GGUF"],"spill":{"mode":"auto","n_gpu_layers_env":null,"gpu_mem_gib":null,"cpu_mem_gib":null,"tensor_split":null,"free_vram_bytes":1543503872}}}`);
  var ENROLL_TOKENS = J(String.raw`[{"id":"9f2c1a7be004","label":"gpu-box-2","created_at":1782010050.12,"revoked":false,"last_used":1782518388.91}]`);
  var PEERS = J(String.raw`[{"name":"hugpy-central","host":"hugpy-central","role":"central","storage_root":"/srv/hugpy/storage","manifest_path":"/srv/hugpy/storage/projects/models_manifest.json","storage_mounted":true,"disk":{"total":858993459200,"used":644245094400,"free":214748364800},"status":"online"},{"name":"gpu-box-1","host":"http://10.8.0.5:9100","role":"worker","storage_root":"http://10.8.0.5:9100","manifest_path":"","storage_mounted":true,"disk":null,"status":"online","gpus":[{"index":0,"name":"NVIDIA GeForce RTX 3090","memory_total":25769803776,"memory_free":1610612736}],"models":["Llama-3.1-8B-Instruct-GGUF","Qwen2.5-Coder-7B-Instruct-GGUF"],"loaded_models":["Qwen2.5-Coder-7B-Instruct-GGUF"],"worker_id":"a1f3c9d2e8b74f60"},{"name":"gpu-box-2","host":"http://10.8.0.6:9100","role":"worker","storage_root":"http://10.8.0.6:9100","manifest_path":"","storage_mounted":true,"disk":null,"status":"online","gpus":[{"index":0,"name":"NVIDIA GeForce RTX 4090","memory_total":25769803776,"memory_free":23622320128}],"models":["Qwen2.5-VL-7B-Instruct"],"loaded_models":[],"worker_id":"b7e2d4a1c6f09e35"}]`);
  var DISCORD_BRIDGES = J(String.raw`[{"id":"b1","mode":"defer","channel_id":"c1","pending":0}]`);
  var DISCORD_CHANNELS = J(String.raw`[{"id":"c1","name":"general","guild":"srv"}]`);
  var DISCORD_MESSAGES = J(String.raw`[{"id":"m1","author":"user","text":"hi","ts":0}]`);
  var DISCORD_BINDINGS = J(String.raw`[{"id":"bind1","channel_id":"c1","user_id":"u1"}]`);
  var DISCORD_USERS = J(String.raw`[{"id":"u1","name":"alice"}]`);
  var SEARCH = J(String.raw`[{"hub_id":"Qwen/Qwen2.5-7B-Instruct","author":"Qwen","downloads":1283400,"likes":1342,"tags":["transformers","safetensors","qwen2","text-generation","conversational","en"],"pipeline_tag":"text-generation","library_name":"transformers","private":false,"total_bytes":15231004672,"last_modified":"2024-09-18T11:04:33+00:00"},{"hub_id":"meta-llama/Llama-3.2-3B-Instruct","author":"meta-llama","downloads":2410773,"likes":984,"tags":["transformers","safetensors","llama","text-generation","conversational","gated"],"pipeline_tag":"text-generation","library_name":"transformers","private":false,"total_bytes":6425499648,"last_modified":"2024-09-25T18:22:10+00:00"},{"hub_id":"bartowski/Qwen2.5-7B-Instruct-GGUF","author":"bartowski","downloads":341208,"likes":214,"tags":["gguf","text-generation","quantized","llama.cpp"],"pipeline_tag":"text-generation","library_name":"gguf","private":false,"total_bytes":31894077440,"last_modified":"2024-09-19T03:41:55+00:00"},{"hub_id":"microsoft/Phi-3.5-mini-instruct","author":"microsoft","downloads":892145,"likes":763,"tags":["transformers","safetensors","phi3","text-generation","conversational"],"pipeline_tag":"text-generation","library_name":"transformers","private":false,"total_bytes":7642236928,"last_modified":"2024-08-20T22:15:02+00:00"},{"hub_id":"TheBloke/Mistral-7B-Instruct-v0.2-GGUF","author":"TheBloke","downloads":541889,"likes":1107,"tags":["gguf","mistral","text-generation","quantized"],"pipeline_tag":"text-generation","library_name":"gguf","private":false,"total_bytes":29135421440,"last_modified":"2023-12-13T09:30:44+00:00"},{"hub_id":"google/gemma-2-2b-it","author":"google","downloads":760312,"likes":645,"tags":["transformers","safetensors","gemma2","text-generation","conversational","gated"],"pipeline_tag":"text-generation","library_name":"transformers","private":false,"total_bytes":5228687360,"last_modified":"2024-08-07T13:50:21+00:00"}]`);
  var HF_SPEC = J(String.raw`{"spec":{"hub_id":"bartowski/Qwen2.5-7B-Instruct-GGUF","license":"apache-2.0","gated":false,"last_modified":"2024-09-19T03:41:55+00:00","total_bytes":31894077440,"num_params":null,"context_length":32768,"gguf_quants":["Q2_K","Q3_K_M","Q4_K_M","Q5_K_M","Q6_K","Q8_0"],"files":[{"path":"Qwen2.5-7B-Instruct-Q2_K.gguf","size":3015940608},{"path":"Qwen2.5-7B-Instruct-Q3_K_M.gguf","size":3808390656},{"path":"Qwen2.5-7B-Instruct-Q4_K_M.gguf","size":4683073024},{"path":"Qwen2.5-7B-Instruct-Q5_K_M.gguf","size":5444831232},{"path":"Qwen2.5-7B-Instruct-Q6_K.gguf","size":6254198272},{"path":"Qwen2.5-7B-Instruct-Q8_0.gguf","size":8098523648},{"path":"config.json","size":661},{"path":"README.md","size":4123}]},"options":{"hub_id":"bartowski/Qwen2.5-7B-Instruct-GGUF","task":"text-generation","options":[{"id":"gguf:Q2_K","label":"GGUF · Q2_K · 2.8 GB","framework":"llama_cpp","filename":"Qwen2.5-7B-Instruct-Q2_K.gguf","include":null,"total_bytes":3015940608,"fits_disk":true},{"id":"gguf:Q3_K_M","label":"GGUF · Q3_K_M · 3.5 GB","framework":"llama_cpp","filename":"Qwen2.5-7B-Instruct-Q3_K_M.gguf","include":null,"total_bytes":3808390656,"fits_disk":true},{"id":"gguf:Q4_K_M","label":"GGUF · Q4_K_M · 4.4 GB","framework":"llama_cpp","filename":"Qwen2.5-7B-Instruct-Q4_K_M.gguf","include":null,"total_bytes":4683073024,"fits_disk":true},{"id":"gguf:Q5_K_M","label":"GGUF · Q5_K_M · 5.1 GB","framework":"llama_cpp","filename":"Qwen2.5-7B-Instruct-Q5_K_M.gguf","include":null,"total_bytes":5444831232,"fits_disk":true},{"id":"gguf:Q6_K","label":"GGUF · Q6_K · 5.8 GB","framework":"llama_cpp","filename":"Qwen2.5-7B-Instruct-Q6_K.gguf","include":null,"total_bytes":6254198272,"fits_disk":true},{"id":"gguf:Q8_0","label":"GGUF · Q8_0 · 7.5 GB","framework":"llama_cpp","filename":"Qwen2.5-7B-Instruct-Q8_0.gguf","include":null,"total_bytes":8098523648,"fits_disk":true}],"recommended":"gguf:Q4_K_M"}}`);
  var PHONES = J(String.raw`[{"id":"9f3c1a2b4d5e6f70","name":"floor-cam-01","host":"10.8.0.21","port":5002,"color":"#58a6ff","url":"http://10.8.0.21:5002","created_at":1751040000.0,"last_seen":1751069988.0,"status":"online","live":{"model_loaded":true,"model_path":"~/phone-brick/ppe-tanishjain-6class.onnx","queue_size":0}},{"id":"1a2b3c4d5e6f7081","name":"gate-cam-02","host":"10.8.0.22","port":5002,"color":"#3fb950","url":"http://10.8.0.22:5002","created_at":1751041200.0,"last_seen":1751069990.0,"status":"online","live":{"model_loaded":true,"model_path":"~/phone-brick/ppe-tanishjain-6class.onnx","queue_size":1}},{"id":"2b3c4d5e6f708192","name":"yard-cam-03","host":"10.8.0.23","port":5002,"color":"#d29922","url":"http://10.8.0.23:5002","created_at":1751042400.0,"last_seen":1751069991.0,"status":"online","live":{"model_loaded":true,"model_path":"~/phone-brick/ppe-tanishjain-6class.onnx","queue_size":0}},{"id":"3c4d5e6f70819203","name":"dock-cam-04","host":"10.8.0.24","port":5002,"color":"#f85149","url":"http://10.8.0.24:5002","created_at":1751043600.0,"last_seen":1751068000.0,"status":"offline","live":{"model_loaded":false,"model_path":null,"queue_size":0}}]`);
  var PHONE_HEALTH = J(String.raw`{"reachable":true,"url":"http://10.8.0.21:5002/status","status":{"model_loaded":true,"queue_size":0,"model_path":"~/phone-brick/ppe-tanishjain-6class.onnx"}}`);
  var PHONE_RUN = J(String.raw`{"id":"e7d6c5b4a3928170","status":"queued","image":"site_floor_2026-06-27.jpg","phone_ids":["9f3c1a2b4d5e6f70","1a2b3c4d5e6f7081","2b3c4d5e6f708192"],"phases":[],"progress":[],"current_phone":null,"cancel_requested":false,"output_rel":null,"error":null,"created_at":1751070000.0,"finished_at":null,"demo":true}`);
  var PHONE_RUN_FRAMES = [
    { delay_ms: 150, event: { type: "status", status: "running" } },
    { delay_ms: 250, event: { type: "current", phone: "floor-cam-01" } },
    { delay_ms: 550, event: { type: "progress", phase: { phone: "floor-cam-01", top_cls: "helmet", top_conf_pct: 91, detections: 4, consensus: null } } },
    { delay_ms: 250, event: { type: "current", phone: "gate-cam-02" } },
    { delay_ms: 550, event: { type: "progress", phase: { phone: "gate-cam-02", top_cls: "no-helmet", top_conf_pct: 78, detections: 2, consensus: null } } },
    { delay_ms: 250, event: { type: "current", phone: "yard-cam-03" } },
    { delay_ms: 550, event: { type: "progress", phase: { phone: "yard-cam-03", top_cls: null, top_conf_pct: 0, detections: 0, consensus: null } } },
    { delay_ms: 200, event: { type: "current", phone: null } },
    { delay_ms: 200, event: { type: "status", status: "done" } },
    { delay_ms: 100, event: { type: "done", run: J(String.raw`{"id":"e7d6c5b4a3928170","status":"done","image":"site_floor_2026-06-27.jpg","phone_ids":["9f3c1a2b4d5e6f70","1a2b3c4d5e6f7081","2b3c4d5e6f708192"],"phases":[{"phone":"floor-cam-01","top_cls":"helmet","top_conf_pct":91,"consensus":"AGR","detections":4,"timestamp":1751070012.4},{"phone":"gate-cam-02","top_cls":"no-helmet","top_conf_pct":78,"consensus":"DIS","detections":2,"timestamp":1751070014.9},{"phone":"yard-cam-03","top_cls":null,"top_conf_pct":0,"consensus":"NOD","detections":0,"timestamp":1751070017.2}],"progress":[{"phone":"floor-cam-01","top_cls":"helmet","top_conf_pct":91,"detections":4,"consensus":null},{"phone":"gate-cam-02","top_cls":"no-helmet","top_conf_pct":78,"detections":2,"consensus":null},{"phone":"yard-cam-03","top_cls":null,"top_conf_pct":0,"detections":0,"consensus":null}],"current_phone":null,"cancel_requested":false,"output_rel":"e7d6c5b4a3928170/annotated_site_floor.jpg","error":null,"created_at":1751070000.0,"finished_at":1751070017.5}`) } }
  ];
  var UPLOAD = { path: "/uploads/demo-attachment.png", name: "demo-attachment.png", size: 12345, demo: true };
  var CHAT_REPLAY = [
    {
      request_id: "req_a1b2c3d4e5",
      frames: [
        { delay_ms: 80, event: { type: "request", request_id: "req_a1b2c3d4e5", stage: "accept" } },
        { delay_ms: 220, event: { type: "status", request_id: "req_a1b2c3d4e5", stage: "dispatch", served_by: "worker", worker_id: "wkr_8f3a21", worker_name: "rtx4090-bench" } },
        { delay_ms: 380, event: { type: "status", request_id: "req_a1b2c3d4e5", stage: "provision", message: "loading Qwen2.5-3B-Instruct-GGUF", progress: 0.62 } },
        { delay_ms: 520, event: { type: "token", text: `Hugpy unifies mixed hardware — a workstation GPU, an idle laptop, even a phone on llama.cpp — behind a single OpenAI-compatible API.` } },
        { delay_ms: 360, event: { type: "token", text: ` A central node holds the model registry and a live worker pool, and routes each chat request at dispatch time to whichever worker has that model assigned,` } },
        { delay_ms: 360, event: { type: "token", text: ` falling back to local inference when none can serve it. Replies stream token-by-token over SSE and auto-continue past any token cap, so long answers are never cut off.` } },
        { delay_ms: 200, event: { type: "done", finish_reason: "stop" } }
      ]
    },
    {
      request_id: "req_b2c3d4e5f6",
      frames: [
        { delay_ms: 80, event: { type: "request", request_id: "req_b2c3d4e5f6", stage: "accept" } },
        { delay_ms: 160, event: { type: "status", request_id: "req_b2c3d4e5f6", stage: "dispatch", served_by: "worker", worker_id: "wkr_8f3a21", worker_name: "rtx4090-bench" } },
        { delay_ms: 300, event: { type: "token", text: `• Cost: reuse hardware you already own — no per-token cloud bills, and idle laptops or phones become capacity instead of e-waste.
` } },
        { delay_ms: 340, event: { type: "token", text: `• Control: models, prompts, and data never leave your network; the gateway is OpenAI-compatible, so existing SDKs and tooling work unchanged.
` } },
        { delay_ms: 340, event: { type: "token", text: `• Resilience: dispatch picks a healthy worker per request and falls back to local inference, so one node dying doesn't take chat down.` } },
        { delay_ms: 200, event: { type: "done", finish_reason: "stop" } }
      ]
    },
    {
      request_id: "req_c3d4e5f6a7",
      frames: [
        { delay_ms: 80, event: { type: "request", request_id: "req_c3d4e5f6a7", stage: "accept" } },
        { delay_ms: 200, event: { type: "status", request_id: "req_c3d4e5f6a7", stage: "dispatch", served_by: "local", worker_id: "", worker_name: "local" } },
        { delay_ms: 460, event: { type: "token", text: `Yes. Hugpy ships a torch-free vision path: llama.cpp serves a GGUF language model alongside its mmproj CLIP projector, so a model like Qwen2.5-VL-3B runs even on CPU-only or phone workers that can't install torch.` } },
        { delay_ms: 320, event: { type: "status", request_id: "req_c3d4e5f6a7", stage: "generate", message: "continuing (part 2)…", segment: 2 } },
        { delay_ms: 420, event: { type: "token", text: ` You attach an image in the chat box — it rides inline as base64 in the request — and the gateway folds it into your latest turn before routing to whichever worker reports vision support, or the local engine here.` } },
        { delay_ms: 200, event: { type: "done", finish_reason: "stop" } }
      ]
    }
  ];
  var DEMO_MSG = "This is a demo — install hugpy (pip install hugpy) and run the console to do this for real.";
  var MUTATING_DEFAULT = { demo: true, ok: true, message: DEMO_MSG };
  var MUTATING = {
    download: J(String.raw`{"id":"demo-job","model_key":"sd-turbo","status":"completed","message":"Demo mode — no real download. Install hugpy to fetch models.","error":null,"progress":1.0,"total_bytes":2580000000,"downloaded_bytes":2580000000,"attempt":1,"max_attempts":4,"stalled":false,"bytes_per_second":null,"created_at":"2026-06-27T17:42:11.103284+00:00","updated_at":"2026-06-27T17:42:12.000000+00:00","demo":true}`),
    repoDownload: J(String.raw`{"id":"demo-job-repo","model_key":"Qwen2.5-7B-Instruct-GGUF","status":"completed","message":"Demo mode — install hugpy to download from Hugging Face.","error":null,"progress":1.0,"total_bytes":4683073024,"downloaded_bytes":4683073024,"attempt":1,"max_attempts":4,"stalled":false,"bytes_per_second":null,"created_at":"2026-06-27T16:40:12.481020+00:00","updated_at":"2026-06-27T16:40:13.000000+00:00","demo":true}`),
    delete: { deleted: false, demo: true, message: "Demo mode — install hugpy to delete model files." },
    prune: { pruned: false, demo: true, message: "Demo mode — install hugpy to prune registry entries." },
    media: { model_key: "Qwen2.5-VL-3B-Instruct-GGUF", media: true, demo: true, message: "Demo mode — media toggle not persisted." },
    jobCancel: { cancelled: false, reason: "demo mode", demo: true, message: "Demo mode — nothing is really downloading." },
    jobRetry: { retried: false, reason: "demo mode", demo: true, message: "Demo mode — install hugpy to resume downloads." },
    keyMint: J(String.raw`{"id":"demo-key","name":"notebook","pool":"","prefix":"hp_demo","created_at":1751068800.0,"last_used":null,"revoked":false,"key":"hp_demo00000000000000000000000000000000000000","demo":true,"message":"Demo key — install hugpy to mint real API keys."}`),
    keyRevoke: { ok: false, demo: true, message: "Demo mode — install hugpy to revoke keys." },
    keyRequire: { require_key: false, demo: true, message: "Demo mode — key enforcement not changeable." },
    mlGate: { require_key: false, demo: true, message: "Demo mode — media gate not changeable." },
    servingSave: J(String.raw`{"key":"Qwen2.5-3B-Instruct-GGUF","mode":"swap","always_on":false,"endpoint":"http://127.0.0.1:8081/v1","model_name":"Qwen2.5-3B-Instruct-GGUF","port":8081,"n_gpu_layers":-1,"threads":8,"ctx_size":8192,"ttl_seconds":900,"override":{"serve_mode":"swap","n_gpu_layers":-1},"available_gguf":["qwen2.5-3b-instruct-q4_k_m.gguf","qwen2.5-3b-instruct-q5_k_m.gguf"],"gguf_file":"qwen2.5-3b-instruct-q4_k_m.gguf","apply":{"applied":false,"reason":"Demo mode — install hugpy to write + restart the systemd unit"},"demo":true}`),
    slotLoad: { loaded: false, reason: "This is a demo — install hugpy to load models into a slot.", slots: [], demo: true },
    slotUnload: { slot_id: "1", model_key: null, healthy: false, demo: true, message: "Demo mode — install hugpy to unload slots." },
    cacheWarm: { ok: true, warming: false, demo: true, message: "Demo mode — install hugpy to warm the SSD cache." },
    freeWorker: { ok: true, demo: true, message: "Demo mode — install hugpy to recycle the API worker." },
    chatCancel: { cancelled: true, demo: true },
    workerUnload: { ok: false, evicted: false, demo: true, message: "Demo mode — install hugpy to free worker VRAM." },
    workerProbe: { ok: true, fit: true, demo: true, message: "Demo mode — fit result is illustrative." },
    enrollMint: J(String.raw`{"id":"demo-token","label":"demo","created_at":1782518400.0,"revoked":false,"last_used":null,"token":"hpw_demo000000000000000000000000000000000000000000000000000000000000","demo":true,"message":"Demo enroll token — install hugpy to enroll real workers."}`)
  };

  // src/showroom/demoFetch.js
  var DEMO_MSG2 = "This is a demo — install hugpy (pip install hugpy) and run the console to do this for real.";
  var listeners = /* @__PURE__ */ new Set();
  function toast(msg) {
    for (const fn of [...listeners]) {
      try {
        fn(msg || DEMO_MSG2);
      } catch {
      }
    }
  }
  var enc = new TextEncoder();
  var jsonResp = (body, status = 200) => new Response(JSON.stringify(body), { status, headers: { "content-type": "application/json" } });
  function pathOf(input) {
    let url = typeof input === "string" ? input : input && input.url || "";
    if (/^https?:\/\//i.test(url)) {
      try {
        const u = new URL(url);
        url = u.pathname + u.search;
      } catch {
      }
    }
    return url;
  }
  function methodOf(input, init) {
    return String(init && init.method || input && typeof input === "object" && input.method || "GET").toUpperCase();
  }
  var chatTurn = 0;
  function chatStreamResponse(init) {
    const turn = CHAT_REPLAY[chatTurn % CHAT_REPLAY.length];
    chatTurn += 1;
    const signal = init && init.signal;
    let cancelled = false;
    let timer = null;
    const stream = new ReadableStream({
      start(controller) {
        const frames = turn.frames;
        let i = 0;
        const step = () => {
          if (cancelled) return;
          if (i >= frames.length) {
            try {
              controller.close();
            } catch {
            }
            return;
          }
          const f = frames[i++];
          timer = setTimeout(() => {
            if (cancelled) return;
            try {
              controller.enqueue(enc.encode("data: " + JSON.stringify(f.event) + "\n\n"));
            } catch {
            }
            step();
          }, f.delay_ms);
        };
        const onAbort = () => {
          cancelled = true;
          if (timer) clearTimeout(timer);
          try {
            controller.close();
          } catch {
          }
        };
        if (signal) {
          if (signal.aborted) {
            onAbort();
            return;
          }
          signal.addEventListener("abort", onAbort, { once: true });
        }
        step();
      },
      cancel() {
        cancelled = true;
        if (timer) clearTimeout(timer);
      }
    });
    return new Response(stream, { status: 200, headers: { "content-type": "text/event-stream", "cache-control": "no-cache" } });
  }
  function getRoute(p) {
    if (p === "/api/auth/config") return AUTH_CONFIG;
    if (p === "/api/version") return VERSION;
    if (p === "/api/readiness") return READINESS;
    if (/^\/api\/auth-svc\/me$/.test(p)) return AUTH_ME;
    if (p === "/api/models") return MODELS;
    if (p === "/api/v1/models") return V1_MODELS;
    if (/^\/api\/models\/[^/]+$/.test(p)) return MODEL_BY_KEY;
    if (p === "/api/ml") return ML;
    if (p === "/api/ml/gate") return ML_GATE;
    if (p === "/api/keys") return KEYS;
    if (p === "/api/prompt/tasks") return PROMPT_TASKS;
    if (/^\/api\/jobs\/[^/]+$/.test(p)) return MUTATING.download;
    if (p === "/api/llm/queue") return QUEUE;
    if (p === "/api/llm/slots") return SLOTS;
    if (p === "/api/llm/slots/install") return SLOTS_INSTALL;
    if (p === "/api/llm/cache") return CACHE;
    if (p === "/api/llm/serving") return SERVING;
    if (/^\/api\/llm\/serving\/[^/]+$/.test(p)) return SERVING_BY_KEY;
    if (p === "/api/llm/workers") return WORKERS;
    if (/^\/api\/llm\/workers\/[^/]+\/health$/.test(p)) return WORKER_HEALTH;
    if (/^\/api\/llm\/workers\/[^/]+$/.test(p)) return WORKERS[0];
    if (p === "/api/llm/enroll-tokens") return ENROLL_TOKENS;
    if (p === "/api/llm/peers") return PEERS;
    if (p.startsWith("/api/search")) return SEARCH;
    if (p.startsWith("/api/hf/spec")) return HF_SPEC;
    if (p === "/api/phone-brick/phones") return PHONES;
    if (/^\/api\/phone-brick\/phones\/[^/]+\/health$/.test(p)) return PHONE_HEALTH;
    if (p === "/api/discord/bridges") return DISCORD_BRIDGES;
    if (p === "/api/discord/channels") return DISCORD_CHANNELS;
    if (/^\/api\/discord\/bridges\/[^/]+\/messages$/.test(p)) return DISCORD_MESSAGES;
    if (p === "/api/discord/bindings") return DISCORD_BINDINGS;
    if (p === "/api/discord/users") return DISCORD_USERS;
    return void 0;
  }
  function mutateRoute(method, p) {
    const M = MUTATING;
    const fleet = "Demo mode — install hugpy to manage the worker fleet.";
    if (method === "POST" && /^\/api\/models\/[^/]+\/download$/.test(p)) return { body: M.download };
    if (method === "POST" && p === "/api/llm/repos/download") return { body: M.repoDownload };
    if (method === "DELETE" && /^\/api\/models\/[^/]+$/.test(p)) return { body: M.delete, msg: M.delete.message };
    if (method === "POST" && /^\/api\/models\/[^/]+\/prune$/.test(p)) return { body: M.prune, msg: M.prune.message };
    if (method === "POST" && /^\/api\/models\/[^/]+\/media$/.test(p)) return { body: M.media };
    if (method === "POST" && /^\/api\/jobs\/[^/]+\/cancel$/.test(p)) return { body: M.jobCancel };
    if (method === "POST" && /^\/api\/jobs\/[^/]+\/retry$/.test(p)) return { body: M.jobRetry };
    if (method === "POST" && p === "/api/uploads") return { body: UPLOAD };
    if (method === "POST" && p === "/api/keys") return { body: M.keyMint, msg: M.keyMint.message };
    if (method === "DELETE" && /^\/api\/keys\/[^/]+$/.test(p)) return { body: M.keyRevoke, msg: M.keyRevoke.message };
    if (method === "PUT" && p === "/api/keys/require") return { body: M.keyRequire, msg: M.keyRequire.message };
    if (method === "PUT" && p === "/api/ml/gate") return { body: M.mlGate, msg: M.mlGate.message };
    if (method === "POST" && /^\/api\/llm\/serving\/[^/]+$/.test(p)) return { body: M.servingSave, msg: "Demo mode — install hugpy to write + restart the serving unit." };
    if (method === "POST" && p === "/api/llm/slots/load") return { body: M.slotLoad, msg: M.slotLoad.reason };
    if (method === "POST" && p === "/api/llm/slots/unload") return { body: M.slotUnload, msg: M.slotUnload.message };
    if (method === "POST" && p === "/api/llm/cache/warm") return { body: M.cacheWarm, msg: M.cacheWarm.message };
    if (method === "POST" && p === "/api/llm/free-worker") return { body: M.freeWorker, msg: M.freeWorker.message };
    if (method === "POST" && /^\/api\/llm\/chat\/cancel\/[^/]+$/.test(p)) return { body: M.chatCancel };
    if (method === "POST" && /^\/api\/llm\/workers\/[^/]+\/probe$/.test(p)) return { body: M.workerProbe, msg: M.workerProbe.message };
    if (method === "POST" && /^\/api\/llm\/workers\/[^/]+\/unload$/.test(p)) return { body: M.workerUnload, msg: M.workerUnload.message };
    if (method === "POST" && /^\/api\/llm\/workers\/[^/]+\/(assign|unassign|admit|block|pool)$/.test(p)) return { body: MUTATING_DEFAULT, msg: fleet };
    if (method === "POST" && p === "/api/llm/workers/register") return { body: { id: "demo-worker", demo: true, message: fleet }, msg: fleet };
    if (method === "DELETE" && /^\/api\/llm\/workers\/[^/]+$/.test(p)) return { body: { ok: false, demo: true, message: "Demo mode — install hugpy to deregister workers." } };
    if (method === "POST" && p === "/api/llm/enroll-tokens") return { body: M.enrollMint, msg: M.enrollMint.message };
    if (method === "DELETE" && /^\/api\/llm\/enroll-tokens\/[^/]+$/.test(p)) return { body: { revoked: false, demo: true, message: "Demo mode — install hugpy to revoke tokens." } };
    if (method === "POST" && p === "/api/discord/bridges") return { body: { id: "demo-bridge", demo: true }, msg: "Demo mode — install hugpy to create Discord bridges." };
    if (method === "DELETE" && /^\/api\/discord\/bridges\/[^/]+$/.test(p)) return { body: { ok: false, demo: true }, msg: "Demo mode — install hugpy to delete bridges." };
    if (method === "POST" && /^\/api\/discord\/bridges\/[^/]+\/(approve|reject|send)$/.test(p)) return { body: { ok: true, demo: true }, msg: "Demo mode — install hugpy to manage Discord." };
    if (method === "POST" && p === "/api/discord/bindings") return { body: { id: "demo-binding", demo: true }, msg: "Demo mode — install hugpy to create bindings." };
    if (method === "DELETE" && /^\/api\/discord\/bindings\/[^/]+$/.test(p)) return { body: { ok: false, demo: true }, msg: "Demo mode — install hugpy to delete bindings." };
    if (method === "POST" && p === "/api/discord/outbox") return { body: { ok: true, demo: true }, msg: "Demo mode — install hugpy to post to Discord." };
    if (method === "POST" && p === "/api/phone-brick/run") return { body: PHONE_RUN };
    if (method === "POST" && /^\/api\/phone-brick\/runs\/[^/]+\/cancel$/.test(p)) return { body: { ok: true, demo: true } };
    if (method === "DELETE" && /^\/api\/phone-brick\/phones\/[^/]+$/.test(p)) return { body: { ok: false, demo: true }, msg: "Demo mode — install hugpy to manage phones." };
    if (/^\/api\/auth-svc\//.test(p)) return { body: { ok: true, demo: true } };
    return void 0;
  }
  async function demoFetch(input, init) {
    const path = pathOf(input);
    const method = methodOf(input, init);
    try {
      if (method === "POST" && /^\/api\/chat\/stream$/.test(path)) return chatStreamResponse(init);
      if (method === "GET") {
        const g2 = getRoute(path);
        if (g2 !== void 0) return jsonResp(g2);
        if (path.startsWith("/api/")) {
          try {
            console.warn("[showroom] unmatched demo GET", path);
          } catch {
          }
          return jsonResp([]);
        }
        return jsonResp({});
      }
      const m = mutateRoute(method, path);
      if (m) {
        if (m.msg) toast(m.msg);
        return jsonResp(m.body);
      }
      try {
        console.warn("[showroom] unmatched demo write", method, path);
      } catch {
      }
      toast(DEMO_MSG2);
      return jsonResp(MUTATING_DEFAULT);
    } catch (e) {
      return jsonResp({ error: "demo shim error", detail: String(e && e.message || e) }, 500);
    }
  }

  // .design-sync/previews/HugpyConsole.tsx
  var import_jsx_runtime = __toESM(require_react_shim(), 1);
  function Console() {
    return /* @__PURE__ */ (0, import_jsx_runtime.jsx)("div", { style: { height: "100vh", width: "100%" }, children: /* @__PURE__ */ (0, import_jsx_runtime.jsx)(ds_exports.HugpyConsole, { fetch: demoFetch }) });
  }
  return __toCommonJS(HugpyConsole_exports);
})();
