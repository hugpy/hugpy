// Phase 3: sourced from the hugpy config module (single source of truth for URLs).
import { hugpyConfig } from "../../../config";

export const FILE_UPLOAD_ENDPOINT = hugpyConfig.uploadUrl;
