export namespace AUTH_CONFIG {
    let mode: string;
    let base: any;
}
export namespace VERSION {
    let name: string;
    let version: string;
    let api: number;
    let auth_mode: string;
}
export const READINESS: any;
export namespace AUTH_ME {
    let authenticated: boolean;
}
export const MODELS: any;
export const MODEL_BY_KEY: any;
export const V1_MODELS: any;
export const ML: any;
export namespace ML_GATE {
    let require_key: boolean;
}
export const PROMPT_TASKS: any;
export const KEYS: any;
export const JOB: any;
export const QUEUE: any;
export const SLOTS: any;
export const CACHE: any;
export const LLM_GROUPS: any;
export const SERVING: any;
export const SERVING_BY_KEY: any;
export namespace SLOTS_INSTALL {
    let steps: any[];
}
export const WORKERS: any;
export const WORKER_HEALTH: any;
export const ENROLL_TOKENS: any;
export const PEERS: any;
export const DISCORD_BRIDGES: any;
export const DISCORD_CHANNELS: any;
export const DISCORD_MESSAGES: any;
export const DISCORD_BINDINGS: any;
export const DISCORD_USERS: any;
export const SEARCH: any;
export const HF_SPEC: any;
export const PHONES: any;
export const PHONE_HEALTH: any;
export const PHONE_RUN: any;
export const PHONE_RUN_FRAMES: ({
    delay_ms: number;
    event: {
        type: string;
        status: string;
        phone?: undefined;
        phase?: undefined;
        run?: undefined;
    };
} | {
    delay_ms: number;
    event: {
        type: string;
        phone: string;
        status?: undefined;
        phase?: undefined;
        run?: undefined;
    };
} | {
    delay_ms: number;
    event: {
        type: string;
        phase: {
            phone: string;
            top_cls: string;
            top_conf_pct: number;
            detections: number;
            consensus: any;
        };
        status?: undefined;
        phone?: undefined;
        run?: undefined;
    };
} | {
    delay_ms: number;
    event: {
        type: string;
        run: any;
        status?: undefined;
        phone?: undefined;
        phase?: undefined;
    };
})[];
export namespace UPLOAD {
    export let path: string;
    let name_1: string;
    export { name_1 as name };
    export let size: number;
    export let demo: boolean;
}
export const JOBS: any;
export const LLM_JOBS: any;
export const CENTRAL_PROVISIONING: any;
export namespace CENTRAL_ADDRESS {
    let lan_ip: string;
    let base_url: string;
}
export namespace HF_AUTH {
    let configured: boolean;
    let valid: boolean;
    let user: any;
}
export const DISCORD_SESSIONS: any;
export const CIVITAI_SEARCH: any[];
export const CIVITAI_DOWNLOADS: {};
export const CHAT_REPLAY: ({
    request_id: string;
    frames: ({
        delay_ms: number;
        event: {
            type: string;
            request_id: string;
            stage: string;
            served_by?: undefined;
            worker_id?: undefined;
            worker_name?: undefined;
            message?: undefined;
            progress?: undefined;
            text?: undefined;
            finish_reason?: undefined;
        };
    } | {
        delay_ms: number;
        event: {
            type: string;
            request_id: string;
            stage: string;
            served_by: string;
            worker_id: string;
            worker_name: string;
            message?: undefined;
            progress?: undefined;
            text?: undefined;
            finish_reason?: undefined;
        };
    } | {
        delay_ms: number;
        event: {
            type: string;
            request_id: string;
            stage: string;
            message: string;
            progress: number;
            served_by?: undefined;
            worker_id?: undefined;
            worker_name?: undefined;
            text?: undefined;
            finish_reason?: undefined;
        };
    } | {
        delay_ms: number;
        event: {
            type: string;
            text: string;
            request_id?: undefined;
            stage?: undefined;
            served_by?: undefined;
            worker_id?: undefined;
            worker_name?: undefined;
            message?: undefined;
            progress?: undefined;
            finish_reason?: undefined;
        };
    } | {
        delay_ms: number;
        event: {
            type: string;
            finish_reason: string;
            request_id?: undefined;
            stage?: undefined;
            served_by?: undefined;
            worker_id?: undefined;
            worker_name?: undefined;
            message?: undefined;
            progress?: undefined;
            text?: undefined;
        };
    })[];
} | {
    request_id: string;
    frames: ({
        delay_ms: number;
        event: {
            type: string;
            request_id: string;
            stage: string;
            served_by?: undefined;
            worker_id?: undefined;
            worker_name?: undefined;
            text?: undefined;
            message?: undefined;
            segment?: undefined;
            finish_reason?: undefined;
        };
    } | {
        delay_ms: number;
        event: {
            type: string;
            request_id: string;
            stage: string;
            served_by: string;
            worker_id: string;
            worker_name: string;
            text?: undefined;
            message?: undefined;
            segment?: undefined;
            finish_reason?: undefined;
        };
    } | {
        delay_ms: number;
        event: {
            type: string;
            text: string;
            request_id?: undefined;
            stage?: undefined;
            served_by?: undefined;
            worker_id?: undefined;
            worker_name?: undefined;
            message?: undefined;
            segment?: undefined;
            finish_reason?: undefined;
        };
    } | {
        delay_ms: number;
        event: {
            type: string;
            request_id: string;
            stage: string;
            message: string;
            segment: number;
            served_by?: undefined;
            worker_id?: undefined;
            worker_name?: undefined;
            text?: undefined;
            finish_reason?: undefined;
        };
    } | {
        delay_ms: number;
        event: {
            type: string;
            finish_reason: string;
            request_id?: undefined;
            stage?: undefined;
            served_by?: undefined;
            worker_id?: undefined;
            worker_name?: undefined;
            text?: undefined;
            message?: undefined;
            segment?: undefined;
        };
    })[];
})[];
export namespace MUTATING_DEFAULT {
    let demo_1: boolean;
    export { demo_1 as demo };
    export let ok: boolean;
    export { DEMO_MSG as message };
}
export namespace MUTATING {
    export let download: any;
    export let repoDownload: any;
    export namespace _delete {
        export let deleted: boolean;
        let demo_2: boolean;
        export { demo_2 as demo };
        export let message: string;
    }
    export { _delete as delete };
    export namespace prune {
        export let pruned: boolean;
        let demo_3: boolean;
        export { demo_3 as demo };
        let message_1: string;
        export { message_1 as message };
    }
    export namespace media {
        export let model_key: string;
        let media_1: boolean;
        export { media_1 as media };
        let demo_4: boolean;
        export { demo_4 as demo };
        let message_2: string;
        export { message_2 as message };
    }
    export namespace jobCancel {
        export let cancelled: boolean;
        export let reason: string;
        let demo_5: boolean;
        export { demo_5 as demo };
        let message_3: string;
        export { message_3 as message };
    }
    export namespace jobRetry {
        export let retried: boolean;
        let reason_1: string;
        export { reason_1 as reason };
        let demo_6: boolean;
        export { demo_6 as demo };
        let message_4: string;
        export { message_4 as message };
    }
    export let keyMint: any;
    export namespace keyRevoke {
        let ok_1: boolean;
        export { ok_1 as ok };
        let demo_7: boolean;
        export { demo_7 as demo };
        let message_5: string;
        export { message_5 as message };
    }
    export namespace keyRequire {
        let require_key_1: boolean;
        export { require_key_1 as require_key };
        let demo_8: boolean;
        export { demo_8 as demo };
        let message_6: string;
        export { message_6 as message };
    }
    export namespace mlGate {
        let require_key_2: boolean;
        export { require_key_2 as require_key };
        let demo_9: boolean;
        export { demo_9 as demo };
        let message_7: string;
        export { message_7 as message };
    }
    export let servingSave: any;
    export namespace slotLoad {
        export let loaded: boolean;
        let reason_2: string;
        export { reason_2 as reason };
        export let slots: any[];
        let demo_10: boolean;
        export { demo_10 as demo };
    }
    export namespace slotUnload {
        export let slot_id: string;
        let model_key_1: any;
        export { model_key_1 as model_key };
        export let healthy: boolean;
        let demo_11: boolean;
        export { demo_11 as demo };
        let message_8: string;
        export { message_8 as message };
    }
    export namespace cacheWarm {
        let ok_2: boolean;
        export { ok_2 as ok };
        export let warming: boolean;
        let demo_12: boolean;
        export { demo_12 as demo };
        let message_9: string;
        export { message_9 as message };
    }
    export namespace freeWorker {
        let ok_3: boolean;
        export { ok_3 as ok };
        let demo_13: boolean;
        export { demo_13 as demo };
        let message_10: string;
        export { message_10 as message };
    }
    export namespace chatCancel {
        let cancelled_1: boolean;
        export { cancelled_1 as cancelled };
        let demo_14: boolean;
        export { demo_14 as demo };
    }
    export namespace workerUnload {
        let ok_4: boolean;
        export { ok_4 as ok };
        export let evicted: boolean;
        let demo_15: boolean;
        export { demo_15 as demo };
        let message_11: string;
        export { message_11 as message };
    }
    export namespace workerProbe {
        let ok_5: boolean;
        export { ok_5 as ok };
        export let fit: boolean;
        let demo_16: boolean;
        export { demo_16 as demo };
        let message_12: string;
        export { message_12 as message };
    }
    export let enrollMint: any;
}
declare const DEMO_MSG: "This is a demo \u2014 install hugpy (pip install hugpy) and run the console to do this for real.";
export {};
