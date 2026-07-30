import type { ProviderStreams } from "../types.ts";
import { lazyApi } from "./lazy.ts";

export const bcgMessagesApi = (): ProviderStreams => lazyApi(() => import("./bcg-messages.ts"));
