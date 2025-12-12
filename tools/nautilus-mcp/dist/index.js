#!/usr/bin/env bun
// @bun
import { createRequire } from "node:module";
var __defProp = Object.defineProperty;
var __export = (target, all) => {
  for (var name in all)
    __defProp(target, name, {
      get: all[name],
      enumerable: true,
      configurable: true,
      set: (newValue) => all[name] = () => newValue
    });
};
var __require = /* @__PURE__ */ createRequire(import.meta.url);

// node_modules/zod/v3/external.js
var exports_external = {};
__export(exports_external, {
  void: () => voidType,
  util: () => util,
  unknown: () => unknownType,
  union: () => unionType,
  undefined: () => undefinedType,
  tuple: () => tupleType,
  transformer: () => effectsType,
  symbol: () => symbolType,
  string: () => stringType,
  strictObject: () => strictObjectType,
  setErrorMap: () => setErrorMap,
  set: () => setType,
  record: () => recordType,
  quotelessJson: () => quotelessJson,
  promise: () => promiseType,
  preprocess: () => preprocessType,
  pipeline: () => pipelineType,
  ostring: () => ostring,
  optional: () => optionalType,
  onumber: () => onumber,
  oboolean: () => oboolean,
  objectUtil: () => objectUtil,
  object: () => objectType,
  number: () => numberType,
  nullable: () => nullableType,
  null: () => nullType,
  never: () => neverType,
  nativeEnum: () => nativeEnumType,
  nan: () => nanType,
  map: () => mapType,
  makeIssue: () => makeIssue,
  literal: () => literalType,
  lazy: () => lazyType,
  late: () => late,
  isValid: () => isValid,
  isDirty: () => isDirty,
  isAsync: () => isAsync,
  isAborted: () => isAborted,
  intersection: () => intersectionType,
  instanceof: () => instanceOfType,
  getParsedType: () => getParsedType,
  getErrorMap: () => getErrorMap,
  function: () => functionType,
  enum: () => enumType,
  effect: () => effectsType,
  discriminatedUnion: () => discriminatedUnionType,
  defaultErrorMap: () => en_default,
  datetimeRegex: () => datetimeRegex,
  date: () => dateType,
  custom: () => custom,
  coerce: () => coerce,
  boolean: () => booleanType,
  bigint: () => bigIntType,
  array: () => arrayType,
  any: () => anyType,
  addIssueToContext: () => addIssueToContext,
  ZodVoid: () => ZodVoid,
  ZodUnknown: () => ZodUnknown,
  ZodUnion: () => ZodUnion,
  ZodUndefined: () => ZodUndefined,
  ZodType: () => ZodType,
  ZodTuple: () => ZodTuple,
  ZodTransformer: () => ZodEffects,
  ZodSymbol: () => ZodSymbol,
  ZodString: () => ZodString,
  ZodSet: () => ZodSet,
  ZodSchema: () => ZodType,
  ZodRecord: () => ZodRecord,
  ZodReadonly: () => ZodReadonly,
  ZodPromise: () => ZodPromise,
  ZodPipeline: () => ZodPipeline,
  ZodParsedType: () => ZodParsedType,
  ZodOptional: () => ZodOptional,
  ZodObject: () => ZodObject,
  ZodNumber: () => ZodNumber,
  ZodNullable: () => ZodNullable,
  ZodNull: () => ZodNull,
  ZodNever: () => ZodNever,
  ZodNativeEnum: () => ZodNativeEnum,
  ZodNaN: () => ZodNaN,
  ZodMap: () => ZodMap,
  ZodLiteral: () => ZodLiteral,
  ZodLazy: () => ZodLazy,
  ZodIssueCode: () => ZodIssueCode,
  ZodIntersection: () => ZodIntersection,
  ZodFunction: () => ZodFunction,
  ZodFirstPartyTypeKind: () => ZodFirstPartyTypeKind,
  ZodError: () => ZodError,
  ZodEnum: () => ZodEnum,
  ZodEffects: () => ZodEffects,
  ZodDiscriminatedUnion: () => ZodDiscriminatedUnion,
  ZodDefault: () => ZodDefault,
  ZodDate: () => ZodDate,
  ZodCatch: () => ZodCatch,
  ZodBranded: () => ZodBranded,
  ZodBoolean: () => ZodBoolean,
  ZodBigInt: () => ZodBigInt,
  ZodArray: () => ZodArray,
  ZodAny: () => ZodAny,
  Schema: () => ZodType,
  ParseStatus: () => ParseStatus,
  OK: () => OK,
  NEVER: () => NEVER,
  INVALID: () => INVALID,
  EMPTY_PATH: () => EMPTY_PATH,
  DIRTY: () => DIRTY,
  BRAND: () => BRAND
});

// node_modules/zod/v3/helpers/util.js
var util;
(function(util2) {
  util2.assertEqual = (_) => {};
  function assertIs(_arg) {}
  util2.assertIs = assertIs;
  function assertNever(_x) {
    throw new Error;
  }
  util2.assertNever = assertNever;
  util2.arrayToEnum = (items) => {
    const obj = {};
    for (const item of items) {
      obj[item] = item;
    }
    return obj;
  };
  util2.getValidEnumValues = (obj) => {
    const validKeys = util2.objectKeys(obj).filter((k) => typeof obj[obj[k]] !== "number");
    const filtered = {};
    for (const k of validKeys) {
      filtered[k] = obj[k];
    }
    return util2.objectValues(filtered);
  };
  util2.objectValues = (obj) => {
    return util2.objectKeys(obj).map(function(e) {
      return obj[e];
    });
  };
  util2.objectKeys = typeof Object.keys === "function" ? (obj) => Object.keys(obj) : (object) => {
    const keys = [];
    for (const key in object) {
      if (Object.prototype.hasOwnProperty.call(object, key)) {
        keys.push(key);
      }
    }
    return keys;
  };
  util2.find = (arr, checker) => {
    for (const item of arr) {
      if (checker(item))
        return item;
    }
    return;
  };
  util2.isInteger = typeof Number.isInteger === "function" ? (val) => Number.isInteger(val) : (val) => typeof val === "number" && Number.isFinite(val) && Math.floor(val) === val;
  function joinValues(array, separator = " | ") {
    return array.map((val) => typeof val === "string" ? `'${val}'` : val).join(separator);
  }
  util2.joinValues = joinValues;
  util2.jsonStringifyReplacer = (_, value) => {
    if (typeof value === "bigint") {
      return value.toString();
    }
    return value;
  };
})(util || (util = {}));
var objectUtil;
(function(objectUtil2) {
  objectUtil2.mergeShapes = (first, second) => {
    return {
      ...first,
      ...second
    };
  };
})(objectUtil || (objectUtil = {}));
var ZodParsedType = util.arrayToEnum([
  "string",
  "nan",
  "number",
  "integer",
  "float",
  "boolean",
  "date",
  "bigint",
  "symbol",
  "function",
  "undefined",
  "null",
  "array",
  "object",
  "unknown",
  "promise",
  "void",
  "never",
  "map",
  "set"
]);
var getParsedType = (data) => {
  const t = typeof data;
  switch (t) {
    case "undefined":
      return ZodParsedType.undefined;
    case "string":
      return ZodParsedType.string;
    case "number":
      return Number.isNaN(data) ? ZodParsedType.nan : ZodParsedType.number;
    case "boolean":
      return ZodParsedType.boolean;
    case "function":
      return ZodParsedType.function;
    case "bigint":
      return ZodParsedType.bigint;
    case "symbol":
      return ZodParsedType.symbol;
    case "object":
      if (Array.isArray(data)) {
        return ZodParsedType.array;
      }
      if (data === null) {
        return ZodParsedType.null;
      }
      if (data.then && typeof data.then === "function" && data.catch && typeof data.catch === "function") {
        return ZodParsedType.promise;
      }
      if (typeof Map !== "undefined" && data instanceof Map) {
        return ZodParsedType.map;
      }
      if (typeof Set !== "undefined" && data instanceof Set) {
        return ZodParsedType.set;
      }
      if (typeof Date !== "undefined" && data instanceof Date) {
        return ZodParsedType.date;
      }
      return ZodParsedType.object;
    default:
      return ZodParsedType.unknown;
  }
};

// node_modules/zod/v3/ZodError.js
var ZodIssueCode = util.arrayToEnum([
  "invalid_type",
  "invalid_literal",
  "custom",
  "invalid_union",
  "invalid_union_discriminator",
  "invalid_enum_value",
  "unrecognized_keys",
  "invalid_arguments",
  "invalid_return_type",
  "invalid_date",
  "invalid_string",
  "too_small",
  "too_big",
  "invalid_intersection_types",
  "not_multiple_of",
  "not_finite"
]);
var quotelessJson = (obj) => {
  const json = JSON.stringify(obj, null, 2);
  return json.replace(/"([^"]+)":/g, "$1:");
};

class ZodError extends Error {
  get errors() {
    return this.issues;
  }
  constructor(issues) {
    super();
    this.issues = [];
    this.addIssue = (sub) => {
      this.issues = [...this.issues, sub];
    };
    this.addIssues = (subs = []) => {
      this.issues = [...this.issues, ...subs];
    };
    const actualProto = new.target.prototype;
    if (Object.setPrototypeOf) {
      Object.setPrototypeOf(this, actualProto);
    } else {
      this.__proto__ = actualProto;
    }
    this.name = "ZodError";
    this.issues = issues;
  }
  format(_mapper) {
    const mapper = _mapper || function(issue) {
      return issue.message;
    };
    const fieldErrors = { _errors: [] };
    const processError = (error) => {
      for (const issue of error.issues) {
        if (issue.code === "invalid_union") {
          issue.unionErrors.map(processError);
        } else if (issue.code === "invalid_return_type") {
          processError(issue.returnTypeError);
        } else if (issue.code === "invalid_arguments") {
          processError(issue.argumentsError);
        } else if (issue.path.length === 0) {
          fieldErrors._errors.push(mapper(issue));
        } else {
          let curr = fieldErrors;
          let i = 0;
          while (i < issue.path.length) {
            const el = issue.path[i];
            const terminal = i === issue.path.length - 1;
            if (!terminal) {
              curr[el] = curr[el] || { _errors: [] };
            } else {
              curr[el] = curr[el] || { _errors: [] };
              curr[el]._errors.push(mapper(issue));
            }
            curr = curr[el];
            i++;
          }
        }
      }
    };
    processError(this);
    return fieldErrors;
  }
  static assert(value) {
    if (!(value instanceof ZodError)) {
      throw new Error(`Not a ZodError: ${value}`);
    }
  }
  toString() {
    return this.message;
  }
  get message() {
    return JSON.stringify(this.issues, util.jsonStringifyReplacer, 2);
  }
  get isEmpty() {
    return this.issues.length === 0;
  }
  flatten(mapper = (issue) => issue.message) {
    const fieldErrors = {};
    const formErrors = [];
    for (const sub of this.issues) {
      if (sub.path.length > 0) {
        const firstEl = sub.path[0];
        fieldErrors[firstEl] = fieldErrors[firstEl] || [];
        fieldErrors[firstEl].push(mapper(sub));
      } else {
        formErrors.push(mapper(sub));
      }
    }
    return { formErrors, fieldErrors };
  }
  get formErrors() {
    return this.flatten();
  }
}
ZodError.create = (issues) => {
  const error = new ZodError(issues);
  return error;
};

// node_modules/zod/v3/locales/en.js
var errorMap = (issue, _ctx) => {
  let message;
  switch (issue.code) {
    case ZodIssueCode.invalid_type:
      if (issue.received === ZodParsedType.undefined) {
        message = "Required";
      } else {
        message = `Expected ${issue.expected}, received ${issue.received}`;
      }
      break;
    case ZodIssueCode.invalid_literal:
      message = `Invalid literal value, expected ${JSON.stringify(issue.expected, util.jsonStringifyReplacer)}`;
      break;
    case ZodIssueCode.unrecognized_keys:
      message = `Unrecognized key(s) in object: ${util.joinValues(issue.keys, ", ")}`;
      break;
    case ZodIssueCode.invalid_union:
      message = `Invalid input`;
      break;
    case ZodIssueCode.invalid_union_discriminator:
      message = `Invalid discriminator value. Expected ${util.joinValues(issue.options)}`;
      break;
    case ZodIssueCode.invalid_enum_value:
      message = `Invalid enum value. Expected ${util.joinValues(issue.options)}, received '${issue.received}'`;
      break;
    case ZodIssueCode.invalid_arguments:
      message = `Invalid function arguments`;
      break;
    case ZodIssueCode.invalid_return_type:
      message = `Invalid function return type`;
      break;
    case ZodIssueCode.invalid_date:
      message = `Invalid date`;
      break;
    case ZodIssueCode.invalid_string:
      if (typeof issue.validation === "object") {
        if ("includes" in issue.validation) {
          message = `Invalid input: must include "${issue.validation.includes}"`;
          if (typeof issue.validation.position === "number") {
            message = `${message} at one or more positions greater than or equal to ${issue.validation.position}`;
          }
        } else if ("startsWith" in issue.validation) {
          message = `Invalid input: must start with "${issue.validation.startsWith}"`;
        } else if ("endsWith" in issue.validation) {
          message = `Invalid input: must end with "${issue.validation.endsWith}"`;
        } else {
          util.assertNever(issue.validation);
        }
      } else if (issue.validation !== "regex") {
        message = `Invalid ${issue.validation}`;
      } else {
        message = "Invalid";
      }
      break;
    case ZodIssueCode.too_small:
      if (issue.type === "array")
        message = `Array must contain ${issue.exact ? "exactly" : issue.inclusive ? `at least` : `more than`} ${issue.minimum} element(s)`;
      else if (issue.type === "string")
        message = `String must contain ${issue.exact ? "exactly" : issue.inclusive ? `at least` : `over`} ${issue.minimum} character(s)`;
      else if (issue.type === "number")
        message = `Number must be ${issue.exact ? `exactly equal to ` : issue.inclusive ? `greater than or equal to ` : `greater than `}${issue.minimum}`;
      else if (issue.type === "bigint")
        message = `Number must be ${issue.exact ? `exactly equal to ` : issue.inclusive ? `greater than or equal to ` : `greater than `}${issue.minimum}`;
      else if (issue.type === "date")
        message = `Date must be ${issue.exact ? `exactly equal to ` : issue.inclusive ? `greater than or equal to ` : `greater than `}${new Date(Number(issue.minimum))}`;
      else
        message = "Invalid input";
      break;
    case ZodIssueCode.too_big:
      if (issue.type === "array")
        message = `Array must contain ${issue.exact ? `exactly` : issue.inclusive ? `at most` : `less than`} ${issue.maximum} element(s)`;
      else if (issue.type === "string")
        message = `String must contain ${issue.exact ? `exactly` : issue.inclusive ? `at most` : `under`} ${issue.maximum} character(s)`;
      else if (issue.type === "number")
        message = `Number must be ${issue.exact ? `exactly` : issue.inclusive ? `less than or equal to` : `less than`} ${issue.maximum}`;
      else if (issue.type === "bigint")
        message = `BigInt must be ${issue.exact ? `exactly` : issue.inclusive ? `less than or equal to` : `less than`} ${issue.maximum}`;
      else if (issue.type === "date")
        message = `Date must be ${issue.exact ? `exactly` : issue.inclusive ? `smaller than or equal to` : `smaller than`} ${new Date(Number(issue.maximum))}`;
      else
        message = "Invalid input";
      break;
    case ZodIssueCode.custom:
      message = `Invalid input`;
      break;
    case ZodIssueCode.invalid_intersection_types:
      message = `Intersection results could not be merged`;
      break;
    case ZodIssueCode.not_multiple_of:
      message = `Number must be a multiple of ${issue.multipleOf}`;
      break;
    case ZodIssueCode.not_finite:
      message = "Number must be finite";
      break;
    default:
      message = _ctx.defaultError;
      util.assertNever(issue);
  }
  return { message };
};
var en_default = errorMap;

// node_modules/zod/v3/errors.js
var overrideErrorMap = en_default;
function setErrorMap(map) {
  overrideErrorMap = map;
}
function getErrorMap() {
  return overrideErrorMap;
}
// node_modules/zod/v3/helpers/parseUtil.js
var makeIssue = (params) => {
  const { data, path, errorMaps, issueData } = params;
  const fullPath = [...path, ...issueData.path || []];
  const fullIssue = {
    ...issueData,
    path: fullPath
  };
  if (issueData.message !== undefined) {
    return {
      ...issueData,
      path: fullPath,
      message: issueData.message
    };
  }
  let errorMessage = "";
  const maps = errorMaps.filter((m) => !!m).slice().reverse();
  for (const map of maps) {
    errorMessage = map(fullIssue, { data, defaultError: errorMessage }).message;
  }
  return {
    ...issueData,
    path: fullPath,
    message: errorMessage
  };
};
var EMPTY_PATH = [];
function addIssueToContext(ctx, issueData) {
  const overrideMap = getErrorMap();
  const issue = makeIssue({
    issueData,
    data: ctx.data,
    path: ctx.path,
    errorMaps: [
      ctx.common.contextualErrorMap,
      ctx.schemaErrorMap,
      overrideMap,
      overrideMap === en_default ? undefined : en_default
    ].filter((x) => !!x)
  });
  ctx.common.issues.push(issue);
}

class ParseStatus {
  constructor() {
    this.value = "valid";
  }
  dirty() {
    if (this.value === "valid")
      this.value = "dirty";
  }
  abort() {
    if (this.value !== "aborted")
      this.value = "aborted";
  }
  static mergeArray(status, results) {
    const arrayValue = [];
    for (const s of results) {
      if (s.status === "aborted")
        return INVALID;
      if (s.status === "dirty")
        status.dirty();
      arrayValue.push(s.value);
    }
    return { status: status.value, value: arrayValue };
  }
  static async mergeObjectAsync(status, pairs) {
    const syncPairs = [];
    for (const pair of pairs) {
      const key = await pair.key;
      const value = await pair.value;
      syncPairs.push({
        key,
        value
      });
    }
    return ParseStatus.mergeObjectSync(status, syncPairs);
  }
  static mergeObjectSync(status, pairs) {
    const finalObject = {};
    for (const pair of pairs) {
      const { key, value } = pair;
      if (key.status === "aborted")
        return INVALID;
      if (value.status === "aborted")
        return INVALID;
      if (key.status === "dirty")
        status.dirty();
      if (value.status === "dirty")
        status.dirty();
      if (key.value !== "__proto__" && (typeof value.value !== "undefined" || pair.alwaysSet)) {
        finalObject[key.value] = value.value;
      }
    }
    return { status: status.value, value: finalObject };
  }
}
var INVALID = Object.freeze({
  status: "aborted"
});
var DIRTY = (value) => ({ status: "dirty", value });
var OK = (value) => ({ status: "valid", value });
var isAborted = (x) => x.status === "aborted";
var isDirty = (x) => x.status === "dirty";
var isValid = (x) => x.status === "valid";
var isAsync = (x) => typeof Promise !== "undefined" && x instanceof Promise;
// node_modules/zod/v3/helpers/errorUtil.js
var errorUtil;
(function(errorUtil2) {
  errorUtil2.errToObj = (message) => typeof message === "string" ? { message } : message || {};
  errorUtil2.toString = (message) => typeof message === "string" ? message : message?.message;
})(errorUtil || (errorUtil = {}));

// node_modules/zod/v3/types.js
class ParseInputLazyPath {
  constructor(parent, value, path, key) {
    this._cachedPath = [];
    this.parent = parent;
    this.data = value;
    this._path = path;
    this._key = key;
  }
  get path() {
    if (!this._cachedPath.length) {
      if (Array.isArray(this._key)) {
        this._cachedPath.push(...this._path, ...this._key);
      } else {
        this._cachedPath.push(...this._path, this._key);
      }
    }
    return this._cachedPath;
  }
}
var handleResult = (ctx, result) => {
  if (isValid(result)) {
    return { success: true, data: result.value };
  } else {
    if (!ctx.common.issues.length) {
      throw new Error("Validation failed but no issues detected.");
    }
    return {
      success: false,
      get error() {
        if (this._error)
          return this._error;
        const error = new ZodError(ctx.common.issues);
        this._error = error;
        return this._error;
      }
    };
  }
};
function processCreateParams(params) {
  if (!params)
    return {};
  const { errorMap: errorMap2, invalid_type_error, required_error, description } = params;
  if (errorMap2 && (invalid_type_error || required_error)) {
    throw new Error(`Can't use "invalid_type_error" or "required_error" in conjunction with custom error map.`);
  }
  if (errorMap2)
    return { errorMap: errorMap2, description };
  const customMap = (iss, ctx) => {
    const { message } = params;
    if (iss.code === "invalid_enum_value") {
      return { message: message ?? ctx.defaultError };
    }
    if (typeof ctx.data === "undefined") {
      return { message: message ?? required_error ?? ctx.defaultError };
    }
    if (iss.code !== "invalid_type")
      return { message: ctx.defaultError };
    return { message: message ?? invalid_type_error ?? ctx.defaultError };
  };
  return { errorMap: customMap, description };
}

class ZodType {
  get description() {
    return this._def.description;
  }
  _getType(input) {
    return getParsedType(input.data);
  }
  _getOrReturnCtx(input, ctx) {
    return ctx || {
      common: input.parent.common,
      data: input.data,
      parsedType: getParsedType(input.data),
      schemaErrorMap: this._def.errorMap,
      path: input.path,
      parent: input.parent
    };
  }
  _processInputParams(input) {
    return {
      status: new ParseStatus,
      ctx: {
        common: input.parent.common,
        data: input.data,
        parsedType: getParsedType(input.data),
        schemaErrorMap: this._def.errorMap,
        path: input.path,
        parent: input.parent
      }
    };
  }
  _parseSync(input) {
    const result = this._parse(input);
    if (isAsync(result)) {
      throw new Error("Synchronous parse encountered promise.");
    }
    return result;
  }
  _parseAsync(input) {
    const result = this._parse(input);
    return Promise.resolve(result);
  }
  parse(data, params) {
    const result = this.safeParse(data, params);
    if (result.success)
      return result.data;
    throw result.error;
  }
  safeParse(data, params) {
    const ctx = {
      common: {
        issues: [],
        async: params?.async ?? false,
        contextualErrorMap: params?.errorMap
      },
      path: params?.path || [],
      schemaErrorMap: this._def.errorMap,
      parent: null,
      data,
      parsedType: getParsedType(data)
    };
    const result = this._parseSync({ data, path: ctx.path, parent: ctx });
    return handleResult(ctx, result);
  }
  "~validate"(data) {
    const ctx = {
      common: {
        issues: [],
        async: !!this["~standard"].async
      },
      path: [],
      schemaErrorMap: this._def.errorMap,
      parent: null,
      data,
      parsedType: getParsedType(data)
    };
    if (!this["~standard"].async) {
      try {
        const result = this._parseSync({ data, path: [], parent: ctx });
        return isValid(result) ? {
          value: result.value
        } : {
          issues: ctx.common.issues
        };
      } catch (err) {
        if (err?.message?.toLowerCase()?.includes("encountered")) {
          this["~standard"].async = true;
        }
        ctx.common = {
          issues: [],
          async: true
        };
      }
    }
    return this._parseAsync({ data, path: [], parent: ctx }).then((result) => isValid(result) ? {
      value: result.value
    } : {
      issues: ctx.common.issues
    });
  }
  async parseAsync(data, params) {
    const result = await this.safeParseAsync(data, params);
    if (result.success)
      return result.data;
    throw result.error;
  }
  async safeParseAsync(data, params) {
    const ctx = {
      common: {
        issues: [],
        contextualErrorMap: params?.errorMap,
        async: true
      },
      path: params?.path || [],
      schemaErrorMap: this._def.errorMap,
      parent: null,
      data,
      parsedType: getParsedType(data)
    };
    const maybeAsyncResult = this._parse({ data, path: ctx.path, parent: ctx });
    const result = await (isAsync(maybeAsyncResult) ? maybeAsyncResult : Promise.resolve(maybeAsyncResult));
    return handleResult(ctx, result);
  }
  refine(check, message) {
    const getIssueProperties = (val) => {
      if (typeof message === "string" || typeof message === "undefined") {
        return { message };
      } else if (typeof message === "function") {
        return message(val);
      } else {
        return message;
      }
    };
    return this._refinement((val, ctx) => {
      const result = check(val);
      const setError = () => ctx.addIssue({
        code: ZodIssueCode.custom,
        ...getIssueProperties(val)
      });
      if (typeof Promise !== "undefined" && result instanceof Promise) {
        return result.then((data) => {
          if (!data) {
            setError();
            return false;
          } else {
            return true;
          }
        });
      }
      if (!result) {
        setError();
        return false;
      } else {
        return true;
      }
    });
  }
  refinement(check, refinementData) {
    return this._refinement((val, ctx) => {
      if (!check(val)) {
        ctx.addIssue(typeof refinementData === "function" ? refinementData(val, ctx) : refinementData);
        return false;
      } else {
        return true;
      }
    });
  }
  _refinement(refinement) {
    return new ZodEffects({
      schema: this,
      typeName: ZodFirstPartyTypeKind.ZodEffects,
      effect: { type: "refinement", refinement }
    });
  }
  superRefine(refinement) {
    return this._refinement(refinement);
  }
  constructor(def) {
    this.spa = this.safeParseAsync;
    this._def = def;
    this.parse = this.parse.bind(this);
    this.safeParse = this.safeParse.bind(this);
    this.parseAsync = this.parseAsync.bind(this);
    this.safeParseAsync = this.safeParseAsync.bind(this);
    this.spa = this.spa.bind(this);
    this.refine = this.refine.bind(this);
    this.refinement = this.refinement.bind(this);
    this.superRefine = this.superRefine.bind(this);
    this.optional = this.optional.bind(this);
    this.nullable = this.nullable.bind(this);
    this.nullish = this.nullish.bind(this);
    this.array = this.array.bind(this);
    this.promise = this.promise.bind(this);
    this.or = this.or.bind(this);
    this.and = this.and.bind(this);
    this.transform = this.transform.bind(this);
    this.brand = this.brand.bind(this);
    this.default = this.default.bind(this);
    this.catch = this.catch.bind(this);
    this.describe = this.describe.bind(this);
    this.pipe = this.pipe.bind(this);
    this.readonly = this.readonly.bind(this);
    this.isNullable = this.isNullable.bind(this);
    this.isOptional = this.isOptional.bind(this);
    this["~standard"] = {
      version: 1,
      vendor: "zod",
      validate: (data) => this["~validate"](data)
    };
  }
  optional() {
    return ZodOptional.create(this, this._def);
  }
  nullable() {
    return ZodNullable.create(this, this._def);
  }
  nullish() {
    return this.nullable().optional();
  }
  array() {
    return ZodArray.create(this);
  }
  promise() {
    return ZodPromise.create(this, this._def);
  }
  or(option) {
    return ZodUnion.create([this, option], this._def);
  }
  and(incoming) {
    return ZodIntersection.create(this, incoming, this._def);
  }
  transform(transform) {
    return new ZodEffects({
      ...processCreateParams(this._def),
      schema: this,
      typeName: ZodFirstPartyTypeKind.ZodEffects,
      effect: { type: "transform", transform }
    });
  }
  default(def) {
    const defaultValueFunc = typeof def === "function" ? def : () => def;
    return new ZodDefault({
      ...processCreateParams(this._def),
      innerType: this,
      defaultValue: defaultValueFunc,
      typeName: ZodFirstPartyTypeKind.ZodDefault
    });
  }
  brand() {
    return new ZodBranded({
      typeName: ZodFirstPartyTypeKind.ZodBranded,
      type: this,
      ...processCreateParams(this._def)
    });
  }
  catch(def) {
    const catchValueFunc = typeof def === "function" ? def : () => def;
    return new ZodCatch({
      ...processCreateParams(this._def),
      innerType: this,
      catchValue: catchValueFunc,
      typeName: ZodFirstPartyTypeKind.ZodCatch
    });
  }
  describe(description) {
    const This = this.constructor;
    return new This({
      ...this._def,
      description
    });
  }
  pipe(target) {
    return ZodPipeline.create(this, target);
  }
  readonly() {
    return ZodReadonly.create(this);
  }
  isOptional() {
    return this.safeParse(undefined).success;
  }
  isNullable() {
    return this.safeParse(null).success;
  }
}
var cuidRegex = /^c[^\s-]{8,}$/i;
var cuid2Regex = /^[0-9a-z]+$/;
var ulidRegex = /^[0-9A-HJKMNP-TV-Z]{26}$/i;
var uuidRegex = /^[0-9a-fA-F]{8}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{4}\b-[0-9a-fA-F]{12}$/i;
var nanoidRegex = /^[a-z0-9_-]{21}$/i;
var jwtRegex = /^[A-Za-z0-9-_]+\.[A-Za-z0-9-_]+\.[A-Za-z0-9-_]*$/;
var durationRegex = /^[-+]?P(?!$)(?:(?:[-+]?\d+Y)|(?:[-+]?\d+[.,]\d+Y$))?(?:(?:[-+]?\d+M)|(?:[-+]?\d+[.,]\d+M$))?(?:(?:[-+]?\d+W)|(?:[-+]?\d+[.,]\d+W$))?(?:(?:[-+]?\d+D)|(?:[-+]?\d+[.,]\d+D$))?(?:T(?=[\d+-])(?:(?:[-+]?\d+H)|(?:[-+]?\d+[.,]\d+H$))?(?:(?:[-+]?\d+M)|(?:[-+]?\d+[.,]\d+M$))?(?:[-+]?\d+(?:[.,]\d+)?S)?)??$/;
var emailRegex = /^(?!\.)(?!.*\.\.)([A-Z0-9_'+\-\.]*)[A-Z0-9_+-]@([A-Z0-9][A-Z0-9\-]*\.)+[A-Z]{2,}$/i;
var _emojiRegex = `^(\\p{Extended_Pictographic}|\\p{Emoji_Component})+$`;
var emojiRegex;
var ipv4Regex = /^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])$/;
var ipv4CidrRegex = /^(?:(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\.){3}(?:25[0-5]|2[0-4][0-9]|1[0-9][0-9]|[1-9][0-9]|[0-9])\/(3[0-2]|[12]?[0-9])$/;
var ipv6Regex = /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))$/;
var ipv6CidrRegex = /^(([0-9a-fA-F]{1,4}:){7,7}[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,7}:|([0-9a-fA-F]{1,4}:){1,6}:[0-9a-fA-F]{1,4}|([0-9a-fA-F]{1,4}:){1,5}(:[0-9a-fA-F]{1,4}){1,2}|([0-9a-fA-F]{1,4}:){1,4}(:[0-9a-fA-F]{1,4}){1,3}|([0-9a-fA-F]{1,4}:){1,3}(:[0-9a-fA-F]{1,4}){1,4}|([0-9a-fA-F]{1,4}:){1,2}(:[0-9a-fA-F]{1,4}){1,5}|[0-9a-fA-F]{1,4}:((:[0-9a-fA-F]{1,4}){1,6})|:((:[0-9a-fA-F]{1,4}){1,7}|:)|fe80:(:[0-9a-fA-F]{0,4}){0,4}%[0-9a-zA-Z]{1,}|::(ffff(:0{1,4}){0,1}:){0,1}((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])|([0-9a-fA-F]{1,4}:){1,4}:((25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9])\.){3,3}(25[0-5]|(2[0-4]|1{0,1}[0-9]){0,1}[0-9]))\/(12[0-8]|1[01][0-9]|[1-9]?[0-9])$/;
var base64Regex = /^([0-9a-zA-Z+/]{4})*(([0-9a-zA-Z+/]{2}==)|([0-9a-zA-Z+/]{3}=))?$/;
var base64urlRegex = /^([0-9a-zA-Z-_]{4})*(([0-9a-zA-Z-_]{2}(==)?)|([0-9a-zA-Z-_]{3}(=)?))?$/;
var dateRegexSource = `((\\d\\d[2468][048]|\\d\\d[13579][26]|\\d\\d0[48]|[02468][048]00|[13579][26]00)-02-29|\\d{4}-((0[13578]|1[02])-(0[1-9]|[12]\\d|3[01])|(0[469]|11)-(0[1-9]|[12]\\d|30)|(02)-(0[1-9]|1\\d|2[0-8])))`;
var dateRegex = new RegExp(`^${dateRegexSource}$`);
function timeRegexSource(args) {
  let secondsRegexSource = `[0-5]\\d`;
  if (args.precision) {
    secondsRegexSource = `${secondsRegexSource}\\.\\d{${args.precision}}`;
  } else if (args.precision == null) {
    secondsRegexSource = `${secondsRegexSource}(\\.\\d+)?`;
  }
  const secondsQuantifier = args.precision ? "+" : "?";
  return `([01]\\d|2[0-3]):[0-5]\\d(:${secondsRegexSource})${secondsQuantifier}`;
}
function timeRegex(args) {
  return new RegExp(`^${timeRegexSource(args)}$`);
}
function datetimeRegex(args) {
  let regex = `${dateRegexSource}T${timeRegexSource(args)}`;
  const opts = [];
  opts.push(args.local ? `Z?` : `Z`);
  if (args.offset)
    opts.push(`([+-]\\d{2}:?\\d{2})`);
  regex = `${regex}(${opts.join("|")})`;
  return new RegExp(`^${regex}$`);
}
function isValidIP(ip, version) {
  if ((version === "v4" || !version) && ipv4Regex.test(ip)) {
    return true;
  }
  if ((version === "v6" || !version) && ipv6Regex.test(ip)) {
    return true;
  }
  return false;
}
function isValidJWT(jwt, alg) {
  if (!jwtRegex.test(jwt))
    return false;
  try {
    const [header] = jwt.split(".");
    if (!header)
      return false;
    const base64 = header.replace(/-/g, "+").replace(/_/g, "/").padEnd(header.length + (4 - header.length % 4) % 4, "=");
    const decoded = JSON.parse(atob(base64));
    if (typeof decoded !== "object" || decoded === null)
      return false;
    if ("typ" in decoded && decoded?.typ !== "JWT")
      return false;
    if (!decoded.alg)
      return false;
    if (alg && decoded.alg !== alg)
      return false;
    return true;
  } catch {
    return false;
  }
}
function isValidCidr(ip, version) {
  if ((version === "v4" || !version) && ipv4CidrRegex.test(ip)) {
    return true;
  }
  if ((version === "v6" || !version) && ipv6CidrRegex.test(ip)) {
    return true;
  }
  return false;
}

class ZodString extends ZodType {
  _parse(input) {
    if (this._def.coerce) {
      input.data = String(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.string) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.string,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    const status = new ParseStatus;
    let ctx = undefined;
    for (const check of this._def.checks) {
      if (check.kind === "min") {
        if (input.data.length < check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            minimum: check.value,
            type: "string",
            inclusive: true,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        if (input.data.length > check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            maximum: check.value,
            type: "string",
            inclusive: true,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "length") {
        const tooBig = input.data.length > check.value;
        const tooSmall = input.data.length < check.value;
        if (tooBig || tooSmall) {
          ctx = this._getOrReturnCtx(input, ctx);
          if (tooBig) {
            addIssueToContext(ctx, {
              code: ZodIssueCode.too_big,
              maximum: check.value,
              type: "string",
              inclusive: true,
              exact: true,
              message: check.message
            });
          } else if (tooSmall) {
            addIssueToContext(ctx, {
              code: ZodIssueCode.too_small,
              minimum: check.value,
              type: "string",
              inclusive: true,
              exact: true,
              message: check.message
            });
          }
          status.dirty();
        }
      } else if (check.kind === "email") {
        if (!emailRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "email",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "emoji") {
        if (!emojiRegex) {
          emojiRegex = new RegExp(_emojiRegex, "u");
        }
        if (!emojiRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "emoji",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "uuid") {
        if (!uuidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "uuid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "nanoid") {
        if (!nanoidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "nanoid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "cuid") {
        if (!cuidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "cuid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "cuid2") {
        if (!cuid2Regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "cuid2",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "ulid") {
        if (!ulidRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "ulid",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "url") {
        try {
          new URL(input.data);
        } catch {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "url",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "regex") {
        check.regex.lastIndex = 0;
        const testResult = check.regex.test(input.data);
        if (!testResult) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "regex",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "trim") {
        input.data = input.data.trim();
      } else if (check.kind === "includes") {
        if (!input.data.includes(check.value, check.position)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: { includes: check.value, position: check.position },
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "toLowerCase") {
        input.data = input.data.toLowerCase();
      } else if (check.kind === "toUpperCase") {
        input.data = input.data.toUpperCase();
      } else if (check.kind === "startsWith") {
        if (!input.data.startsWith(check.value)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: { startsWith: check.value },
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "endsWith") {
        if (!input.data.endsWith(check.value)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: { endsWith: check.value },
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "datetime") {
        const regex = datetimeRegex(check);
        if (!regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: "datetime",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "date") {
        const regex = dateRegex;
        if (!regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: "date",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "time") {
        const regex = timeRegex(check);
        if (!regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_string,
            validation: "time",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "duration") {
        if (!durationRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "duration",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "ip") {
        if (!isValidIP(input.data, check.version)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "ip",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "jwt") {
        if (!isValidJWT(input.data, check.alg)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "jwt",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "cidr") {
        if (!isValidCidr(input.data, check.version)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "cidr",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "base64") {
        if (!base64Regex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "base64",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "base64url") {
        if (!base64urlRegex.test(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            validation: "base64url",
            code: ZodIssueCode.invalid_string,
            message: check.message
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return { status: status.value, value: input.data };
  }
  _regex(regex, validation, message) {
    return this.refinement((data) => regex.test(data), {
      validation,
      code: ZodIssueCode.invalid_string,
      ...errorUtil.errToObj(message)
    });
  }
  _addCheck(check) {
    return new ZodString({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  email(message) {
    return this._addCheck({ kind: "email", ...errorUtil.errToObj(message) });
  }
  url(message) {
    return this._addCheck({ kind: "url", ...errorUtil.errToObj(message) });
  }
  emoji(message) {
    return this._addCheck({ kind: "emoji", ...errorUtil.errToObj(message) });
  }
  uuid(message) {
    return this._addCheck({ kind: "uuid", ...errorUtil.errToObj(message) });
  }
  nanoid(message) {
    return this._addCheck({ kind: "nanoid", ...errorUtil.errToObj(message) });
  }
  cuid(message) {
    return this._addCheck({ kind: "cuid", ...errorUtil.errToObj(message) });
  }
  cuid2(message) {
    return this._addCheck({ kind: "cuid2", ...errorUtil.errToObj(message) });
  }
  ulid(message) {
    return this._addCheck({ kind: "ulid", ...errorUtil.errToObj(message) });
  }
  base64(message) {
    return this._addCheck({ kind: "base64", ...errorUtil.errToObj(message) });
  }
  base64url(message) {
    return this._addCheck({
      kind: "base64url",
      ...errorUtil.errToObj(message)
    });
  }
  jwt(options) {
    return this._addCheck({ kind: "jwt", ...errorUtil.errToObj(options) });
  }
  ip(options) {
    return this._addCheck({ kind: "ip", ...errorUtil.errToObj(options) });
  }
  cidr(options) {
    return this._addCheck({ kind: "cidr", ...errorUtil.errToObj(options) });
  }
  datetime(options) {
    if (typeof options === "string") {
      return this._addCheck({
        kind: "datetime",
        precision: null,
        offset: false,
        local: false,
        message: options
      });
    }
    return this._addCheck({
      kind: "datetime",
      precision: typeof options?.precision === "undefined" ? null : options?.precision,
      offset: options?.offset ?? false,
      local: options?.local ?? false,
      ...errorUtil.errToObj(options?.message)
    });
  }
  date(message) {
    return this._addCheck({ kind: "date", message });
  }
  time(options) {
    if (typeof options === "string") {
      return this._addCheck({
        kind: "time",
        precision: null,
        message: options
      });
    }
    return this._addCheck({
      kind: "time",
      precision: typeof options?.precision === "undefined" ? null : options?.precision,
      ...errorUtil.errToObj(options?.message)
    });
  }
  duration(message) {
    return this._addCheck({ kind: "duration", ...errorUtil.errToObj(message) });
  }
  regex(regex, message) {
    return this._addCheck({
      kind: "regex",
      regex,
      ...errorUtil.errToObj(message)
    });
  }
  includes(value, options) {
    return this._addCheck({
      kind: "includes",
      value,
      position: options?.position,
      ...errorUtil.errToObj(options?.message)
    });
  }
  startsWith(value, message) {
    return this._addCheck({
      kind: "startsWith",
      value,
      ...errorUtil.errToObj(message)
    });
  }
  endsWith(value, message) {
    return this._addCheck({
      kind: "endsWith",
      value,
      ...errorUtil.errToObj(message)
    });
  }
  min(minLength, message) {
    return this._addCheck({
      kind: "min",
      value: minLength,
      ...errorUtil.errToObj(message)
    });
  }
  max(maxLength, message) {
    return this._addCheck({
      kind: "max",
      value: maxLength,
      ...errorUtil.errToObj(message)
    });
  }
  length(len, message) {
    return this._addCheck({
      kind: "length",
      value: len,
      ...errorUtil.errToObj(message)
    });
  }
  nonempty(message) {
    return this.min(1, errorUtil.errToObj(message));
  }
  trim() {
    return new ZodString({
      ...this._def,
      checks: [...this._def.checks, { kind: "trim" }]
    });
  }
  toLowerCase() {
    return new ZodString({
      ...this._def,
      checks: [...this._def.checks, { kind: "toLowerCase" }]
    });
  }
  toUpperCase() {
    return new ZodString({
      ...this._def,
      checks: [...this._def.checks, { kind: "toUpperCase" }]
    });
  }
  get isDatetime() {
    return !!this._def.checks.find((ch) => ch.kind === "datetime");
  }
  get isDate() {
    return !!this._def.checks.find((ch) => ch.kind === "date");
  }
  get isTime() {
    return !!this._def.checks.find((ch) => ch.kind === "time");
  }
  get isDuration() {
    return !!this._def.checks.find((ch) => ch.kind === "duration");
  }
  get isEmail() {
    return !!this._def.checks.find((ch) => ch.kind === "email");
  }
  get isURL() {
    return !!this._def.checks.find((ch) => ch.kind === "url");
  }
  get isEmoji() {
    return !!this._def.checks.find((ch) => ch.kind === "emoji");
  }
  get isUUID() {
    return !!this._def.checks.find((ch) => ch.kind === "uuid");
  }
  get isNANOID() {
    return !!this._def.checks.find((ch) => ch.kind === "nanoid");
  }
  get isCUID() {
    return !!this._def.checks.find((ch) => ch.kind === "cuid");
  }
  get isCUID2() {
    return !!this._def.checks.find((ch) => ch.kind === "cuid2");
  }
  get isULID() {
    return !!this._def.checks.find((ch) => ch.kind === "ulid");
  }
  get isIP() {
    return !!this._def.checks.find((ch) => ch.kind === "ip");
  }
  get isCIDR() {
    return !!this._def.checks.find((ch) => ch.kind === "cidr");
  }
  get isBase64() {
    return !!this._def.checks.find((ch) => ch.kind === "base64");
  }
  get isBase64url() {
    return !!this._def.checks.find((ch) => ch.kind === "base64url");
  }
  get minLength() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min;
  }
  get maxLength() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max;
  }
}
ZodString.create = (params) => {
  return new ZodString({
    checks: [],
    typeName: ZodFirstPartyTypeKind.ZodString,
    coerce: params?.coerce ?? false,
    ...processCreateParams(params)
  });
};
function floatSafeRemainder(val, step) {
  const valDecCount = (val.toString().split(".")[1] || "").length;
  const stepDecCount = (step.toString().split(".")[1] || "").length;
  const decCount = valDecCount > stepDecCount ? valDecCount : stepDecCount;
  const valInt = Number.parseInt(val.toFixed(decCount).replace(".", ""));
  const stepInt = Number.parseInt(step.toFixed(decCount).replace(".", ""));
  return valInt % stepInt / 10 ** decCount;
}

class ZodNumber extends ZodType {
  constructor() {
    super(...arguments);
    this.min = this.gte;
    this.max = this.lte;
    this.step = this.multipleOf;
  }
  _parse(input) {
    if (this._def.coerce) {
      input.data = Number(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.number) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.number,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    let ctx = undefined;
    const status = new ParseStatus;
    for (const check of this._def.checks) {
      if (check.kind === "int") {
        if (!util.isInteger(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.invalid_type,
            expected: "integer",
            received: "float",
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "min") {
        const tooSmall = check.inclusive ? input.data < check.value : input.data <= check.value;
        if (tooSmall) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            minimum: check.value,
            type: "number",
            inclusive: check.inclusive,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        const tooBig = check.inclusive ? input.data > check.value : input.data >= check.value;
        if (tooBig) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            maximum: check.value,
            type: "number",
            inclusive: check.inclusive,
            exact: false,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "multipleOf") {
        if (floatSafeRemainder(input.data, check.value) !== 0) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.not_multiple_of,
            multipleOf: check.value,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "finite") {
        if (!Number.isFinite(input.data)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.not_finite,
            message: check.message
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return { status: status.value, value: input.data };
  }
  gte(value, message) {
    return this.setLimit("min", value, true, errorUtil.toString(message));
  }
  gt(value, message) {
    return this.setLimit("min", value, false, errorUtil.toString(message));
  }
  lte(value, message) {
    return this.setLimit("max", value, true, errorUtil.toString(message));
  }
  lt(value, message) {
    return this.setLimit("max", value, false, errorUtil.toString(message));
  }
  setLimit(kind, value, inclusive, message) {
    return new ZodNumber({
      ...this._def,
      checks: [
        ...this._def.checks,
        {
          kind,
          value,
          inclusive,
          message: errorUtil.toString(message)
        }
      ]
    });
  }
  _addCheck(check) {
    return new ZodNumber({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  int(message) {
    return this._addCheck({
      kind: "int",
      message: errorUtil.toString(message)
    });
  }
  positive(message) {
    return this._addCheck({
      kind: "min",
      value: 0,
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  negative(message) {
    return this._addCheck({
      kind: "max",
      value: 0,
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  nonpositive(message) {
    return this._addCheck({
      kind: "max",
      value: 0,
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  nonnegative(message) {
    return this._addCheck({
      kind: "min",
      value: 0,
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  multipleOf(value, message) {
    return this._addCheck({
      kind: "multipleOf",
      value,
      message: errorUtil.toString(message)
    });
  }
  finite(message) {
    return this._addCheck({
      kind: "finite",
      message: errorUtil.toString(message)
    });
  }
  safe(message) {
    return this._addCheck({
      kind: "min",
      inclusive: true,
      value: Number.MIN_SAFE_INTEGER,
      message: errorUtil.toString(message)
    })._addCheck({
      kind: "max",
      inclusive: true,
      value: Number.MAX_SAFE_INTEGER,
      message: errorUtil.toString(message)
    });
  }
  get minValue() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min;
  }
  get maxValue() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max;
  }
  get isInt() {
    return !!this._def.checks.find((ch) => ch.kind === "int" || ch.kind === "multipleOf" && util.isInteger(ch.value));
  }
  get isFinite() {
    let max = null;
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "finite" || ch.kind === "int" || ch.kind === "multipleOf") {
        return true;
      } else if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      } else if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return Number.isFinite(min) && Number.isFinite(max);
  }
}
ZodNumber.create = (params) => {
  return new ZodNumber({
    checks: [],
    typeName: ZodFirstPartyTypeKind.ZodNumber,
    coerce: params?.coerce || false,
    ...processCreateParams(params)
  });
};

class ZodBigInt extends ZodType {
  constructor() {
    super(...arguments);
    this.min = this.gte;
    this.max = this.lte;
  }
  _parse(input) {
    if (this._def.coerce) {
      try {
        input.data = BigInt(input.data);
      } catch {
        return this._getInvalidInput(input);
      }
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.bigint) {
      return this._getInvalidInput(input);
    }
    let ctx = undefined;
    const status = new ParseStatus;
    for (const check of this._def.checks) {
      if (check.kind === "min") {
        const tooSmall = check.inclusive ? input.data < check.value : input.data <= check.value;
        if (tooSmall) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            type: "bigint",
            minimum: check.value,
            inclusive: check.inclusive,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        const tooBig = check.inclusive ? input.data > check.value : input.data >= check.value;
        if (tooBig) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            type: "bigint",
            maximum: check.value,
            inclusive: check.inclusive,
            message: check.message
          });
          status.dirty();
        }
      } else if (check.kind === "multipleOf") {
        if (input.data % check.value !== BigInt(0)) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.not_multiple_of,
            multipleOf: check.value,
            message: check.message
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return { status: status.value, value: input.data };
  }
  _getInvalidInput(input) {
    const ctx = this._getOrReturnCtx(input);
    addIssueToContext(ctx, {
      code: ZodIssueCode.invalid_type,
      expected: ZodParsedType.bigint,
      received: ctx.parsedType
    });
    return INVALID;
  }
  gte(value, message) {
    return this.setLimit("min", value, true, errorUtil.toString(message));
  }
  gt(value, message) {
    return this.setLimit("min", value, false, errorUtil.toString(message));
  }
  lte(value, message) {
    return this.setLimit("max", value, true, errorUtil.toString(message));
  }
  lt(value, message) {
    return this.setLimit("max", value, false, errorUtil.toString(message));
  }
  setLimit(kind, value, inclusive, message) {
    return new ZodBigInt({
      ...this._def,
      checks: [
        ...this._def.checks,
        {
          kind,
          value,
          inclusive,
          message: errorUtil.toString(message)
        }
      ]
    });
  }
  _addCheck(check) {
    return new ZodBigInt({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  positive(message) {
    return this._addCheck({
      kind: "min",
      value: BigInt(0),
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  negative(message) {
    return this._addCheck({
      kind: "max",
      value: BigInt(0),
      inclusive: false,
      message: errorUtil.toString(message)
    });
  }
  nonpositive(message) {
    return this._addCheck({
      kind: "max",
      value: BigInt(0),
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  nonnegative(message) {
    return this._addCheck({
      kind: "min",
      value: BigInt(0),
      inclusive: true,
      message: errorUtil.toString(message)
    });
  }
  multipleOf(value, message) {
    return this._addCheck({
      kind: "multipleOf",
      value,
      message: errorUtil.toString(message)
    });
  }
  get minValue() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min;
  }
  get maxValue() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max;
  }
}
ZodBigInt.create = (params) => {
  return new ZodBigInt({
    checks: [],
    typeName: ZodFirstPartyTypeKind.ZodBigInt,
    coerce: params?.coerce ?? false,
    ...processCreateParams(params)
  });
};

class ZodBoolean extends ZodType {
  _parse(input) {
    if (this._def.coerce) {
      input.data = Boolean(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.boolean) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.boolean,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
}
ZodBoolean.create = (params) => {
  return new ZodBoolean({
    typeName: ZodFirstPartyTypeKind.ZodBoolean,
    coerce: params?.coerce || false,
    ...processCreateParams(params)
  });
};

class ZodDate extends ZodType {
  _parse(input) {
    if (this._def.coerce) {
      input.data = new Date(input.data);
    }
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.date) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.date,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    if (Number.isNaN(input.data.getTime())) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_date
      });
      return INVALID;
    }
    const status = new ParseStatus;
    let ctx = undefined;
    for (const check of this._def.checks) {
      if (check.kind === "min") {
        if (input.data.getTime() < check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_small,
            message: check.message,
            inclusive: true,
            exact: false,
            minimum: check.value,
            type: "date"
          });
          status.dirty();
        }
      } else if (check.kind === "max") {
        if (input.data.getTime() > check.value) {
          ctx = this._getOrReturnCtx(input, ctx);
          addIssueToContext(ctx, {
            code: ZodIssueCode.too_big,
            message: check.message,
            inclusive: true,
            exact: false,
            maximum: check.value,
            type: "date"
          });
          status.dirty();
        }
      } else {
        util.assertNever(check);
      }
    }
    return {
      status: status.value,
      value: new Date(input.data.getTime())
    };
  }
  _addCheck(check) {
    return new ZodDate({
      ...this._def,
      checks: [...this._def.checks, check]
    });
  }
  min(minDate, message) {
    return this._addCheck({
      kind: "min",
      value: minDate.getTime(),
      message: errorUtil.toString(message)
    });
  }
  max(maxDate, message) {
    return this._addCheck({
      kind: "max",
      value: maxDate.getTime(),
      message: errorUtil.toString(message)
    });
  }
  get minDate() {
    let min = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "min") {
        if (min === null || ch.value > min)
          min = ch.value;
      }
    }
    return min != null ? new Date(min) : null;
  }
  get maxDate() {
    let max = null;
    for (const ch of this._def.checks) {
      if (ch.kind === "max") {
        if (max === null || ch.value < max)
          max = ch.value;
      }
    }
    return max != null ? new Date(max) : null;
  }
}
ZodDate.create = (params) => {
  return new ZodDate({
    checks: [],
    coerce: params?.coerce || false,
    typeName: ZodFirstPartyTypeKind.ZodDate,
    ...processCreateParams(params)
  });
};

class ZodSymbol extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.symbol) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.symbol,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
}
ZodSymbol.create = (params) => {
  return new ZodSymbol({
    typeName: ZodFirstPartyTypeKind.ZodSymbol,
    ...processCreateParams(params)
  });
};

class ZodUndefined extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.undefined) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.undefined,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
}
ZodUndefined.create = (params) => {
  return new ZodUndefined({
    typeName: ZodFirstPartyTypeKind.ZodUndefined,
    ...processCreateParams(params)
  });
};

class ZodNull extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.null) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.null,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
}
ZodNull.create = (params) => {
  return new ZodNull({
    typeName: ZodFirstPartyTypeKind.ZodNull,
    ...processCreateParams(params)
  });
};

class ZodAny extends ZodType {
  constructor() {
    super(...arguments);
    this._any = true;
  }
  _parse(input) {
    return OK(input.data);
  }
}
ZodAny.create = (params) => {
  return new ZodAny({
    typeName: ZodFirstPartyTypeKind.ZodAny,
    ...processCreateParams(params)
  });
};

class ZodUnknown extends ZodType {
  constructor() {
    super(...arguments);
    this._unknown = true;
  }
  _parse(input) {
    return OK(input.data);
  }
}
ZodUnknown.create = (params) => {
  return new ZodUnknown({
    typeName: ZodFirstPartyTypeKind.ZodUnknown,
    ...processCreateParams(params)
  });
};

class ZodNever extends ZodType {
  _parse(input) {
    const ctx = this._getOrReturnCtx(input);
    addIssueToContext(ctx, {
      code: ZodIssueCode.invalid_type,
      expected: ZodParsedType.never,
      received: ctx.parsedType
    });
    return INVALID;
  }
}
ZodNever.create = (params) => {
  return new ZodNever({
    typeName: ZodFirstPartyTypeKind.ZodNever,
    ...processCreateParams(params)
  });
};

class ZodVoid extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.undefined) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.void,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return OK(input.data);
  }
}
ZodVoid.create = (params) => {
  return new ZodVoid({
    typeName: ZodFirstPartyTypeKind.ZodVoid,
    ...processCreateParams(params)
  });
};

class ZodArray extends ZodType {
  _parse(input) {
    const { ctx, status } = this._processInputParams(input);
    const def = this._def;
    if (ctx.parsedType !== ZodParsedType.array) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.array,
        received: ctx.parsedType
      });
      return INVALID;
    }
    if (def.exactLength !== null) {
      const tooBig = ctx.data.length > def.exactLength.value;
      const tooSmall = ctx.data.length < def.exactLength.value;
      if (tooBig || tooSmall) {
        addIssueToContext(ctx, {
          code: tooBig ? ZodIssueCode.too_big : ZodIssueCode.too_small,
          minimum: tooSmall ? def.exactLength.value : undefined,
          maximum: tooBig ? def.exactLength.value : undefined,
          type: "array",
          inclusive: true,
          exact: true,
          message: def.exactLength.message
        });
        status.dirty();
      }
    }
    if (def.minLength !== null) {
      if (ctx.data.length < def.minLength.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_small,
          minimum: def.minLength.value,
          type: "array",
          inclusive: true,
          exact: false,
          message: def.minLength.message
        });
        status.dirty();
      }
    }
    if (def.maxLength !== null) {
      if (ctx.data.length > def.maxLength.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_big,
          maximum: def.maxLength.value,
          type: "array",
          inclusive: true,
          exact: false,
          message: def.maxLength.message
        });
        status.dirty();
      }
    }
    if (ctx.common.async) {
      return Promise.all([...ctx.data].map((item, i) => {
        return def.type._parseAsync(new ParseInputLazyPath(ctx, item, ctx.path, i));
      })).then((result2) => {
        return ParseStatus.mergeArray(status, result2);
      });
    }
    const result = [...ctx.data].map((item, i) => {
      return def.type._parseSync(new ParseInputLazyPath(ctx, item, ctx.path, i));
    });
    return ParseStatus.mergeArray(status, result);
  }
  get element() {
    return this._def.type;
  }
  min(minLength, message) {
    return new ZodArray({
      ...this._def,
      minLength: { value: minLength, message: errorUtil.toString(message) }
    });
  }
  max(maxLength, message) {
    return new ZodArray({
      ...this._def,
      maxLength: { value: maxLength, message: errorUtil.toString(message) }
    });
  }
  length(len, message) {
    return new ZodArray({
      ...this._def,
      exactLength: { value: len, message: errorUtil.toString(message) }
    });
  }
  nonempty(message) {
    return this.min(1, message);
  }
}
ZodArray.create = (schema, params) => {
  return new ZodArray({
    type: schema,
    minLength: null,
    maxLength: null,
    exactLength: null,
    typeName: ZodFirstPartyTypeKind.ZodArray,
    ...processCreateParams(params)
  });
};
function deepPartialify(schema) {
  if (schema instanceof ZodObject) {
    const newShape = {};
    for (const key in schema.shape) {
      const fieldSchema = schema.shape[key];
      newShape[key] = ZodOptional.create(deepPartialify(fieldSchema));
    }
    return new ZodObject({
      ...schema._def,
      shape: () => newShape
    });
  } else if (schema instanceof ZodArray) {
    return new ZodArray({
      ...schema._def,
      type: deepPartialify(schema.element)
    });
  } else if (schema instanceof ZodOptional) {
    return ZodOptional.create(deepPartialify(schema.unwrap()));
  } else if (schema instanceof ZodNullable) {
    return ZodNullable.create(deepPartialify(schema.unwrap()));
  } else if (schema instanceof ZodTuple) {
    return ZodTuple.create(schema.items.map((item) => deepPartialify(item)));
  } else {
    return schema;
  }
}

class ZodObject extends ZodType {
  constructor() {
    super(...arguments);
    this._cached = null;
    this.nonstrict = this.passthrough;
    this.augment = this.extend;
  }
  _getCached() {
    if (this._cached !== null)
      return this._cached;
    const shape = this._def.shape();
    const keys = util.objectKeys(shape);
    this._cached = { shape, keys };
    return this._cached;
  }
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.object) {
      const ctx2 = this._getOrReturnCtx(input);
      addIssueToContext(ctx2, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.object,
        received: ctx2.parsedType
      });
      return INVALID;
    }
    const { status, ctx } = this._processInputParams(input);
    const { shape, keys: shapeKeys } = this._getCached();
    const extraKeys = [];
    if (!(this._def.catchall instanceof ZodNever && this._def.unknownKeys === "strip")) {
      for (const key in ctx.data) {
        if (!shapeKeys.includes(key)) {
          extraKeys.push(key);
        }
      }
    }
    const pairs = [];
    for (const key of shapeKeys) {
      const keyValidator = shape[key];
      const value = ctx.data[key];
      pairs.push({
        key: { status: "valid", value: key },
        value: keyValidator._parse(new ParseInputLazyPath(ctx, value, ctx.path, key)),
        alwaysSet: key in ctx.data
      });
    }
    if (this._def.catchall instanceof ZodNever) {
      const unknownKeys = this._def.unknownKeys;
      if (unknownKeys === "passthrough") {
        for (const key of extraKeys) {
          pairs.push({
            key: { status: "valid", value: key },
            value: { status: "valid", value: ctx.data[key] }
          });
        }
      } else if (unknownKeys === "strict") {
        if (extraKeys.length > 0) {
          addIssueToContext(ctx, {
            code: ZodIssueCode.unrecognized_keys,
            keys: extraKeys
          });
          status.dirty();
        }
      } else if (unknownKeys === "strip") {} else {
        throw new Error(`Internal ZodObject error: invalid unknownKeys value.`);
      }
    } else {
      const catchall = this._def.catchall;
      for (const key of extraKeys) {
        const value = ctx.data[key];
        pairs.push({
          key: { status: "valid", value: key },
          value: catchall._parse(new ParseInputLazyPath(ctx, value, ctx.path, key)),
          alwaysSet: key in ctx.data
        });
      }
    }
    if (ctx.common.async) {
      return Promise.resolve().then(async () => {
        const syncPairs = [];
        for (const pair of pairs) {
          const key = await pair.key;
          const value = await pair.value;
          syncPairs.push({
            key,
            value,
            alwaysSet: pair.alwaysSet
          });
        }
        return syncPairs;
      }).then((syncPairs) => {
        return ParseStatus.mergeObjectSync(status, syncPairs);
      });
    } else {
      return ParseStatus.mergeObjectSync(status, pairs);
    }
  }
  get shape() {
    return this._def.shape();
  }
  strict(message) {
    errorUtil.errToObj;
    return new ZodObject({
      ...this._def,
      unknownKeys: "strict",
      ...message !== undefined ? {
        errorMap: (issue, ctx) => {
          const defaultError = this._def.errorMap?.(issue, ctx).message ?? ctx.defaultError;
          if (issue.code === "unrecognized_keys")
            return {
              message: errorUtil.errToObj(message).message ?? defaultError
            };
          return {
            message: defaultError
          };
        }
      } : {}
    });
  }
  strip() {
    return new ZodObject({
      ...this._def,
      unknownKeys: "strip"
    });
  }
  passthrough() {
    return new ZodObject({
      ...this._def,
      unknownKeys: "passthrough"
    });
  }
  extend(augmentation) {
    return new ZodObject({
      ...this._def,
      shape: () => ({
        ...this._def.shape(),
        ...augmentation
      })
    });
  }
  merge(merging) {
    const merged = new ZodObject({
      unknownKeys: merging._def.unknownKeys,
      catchall: merging._def.catchall,
      shape: () => ({
        ...this._def.shape(),
        ...merging._def.shape()
      }),
      typeName: ZodFirstPartyTypeKind.ZodObject
    });
    return merged;
  }
  setKey(key, schema) {
    return this.augment({ [key]: schema });
  }
  catchall(index) {
    return new ZodObject({
      ...this._def,
      catchall: index
    });
  }
  pick(mask) {
    const shape = {};
    for (const key of util.objectKeys(mask)) {
      if (mask[key] && this.shape[key]) {
        shape[key] = this.shape[key];
      }
    }
    return new ZodObject({
      ...this._def,
      shape: () => shape
    });
  }
  omit(mask) {
    const shape = {};
    for (const key of util.objectKeys(this.shape)) {
      if (!mask[key]) {
        shape[key] = this.shape[key];
      }
    }
    return new ZodObject({
      ...this._def,
      shape: () => shape
    });
  }
  deepPartial() {
    return deepPartialify(this);
  }
  partial(mask) {
    const newShape = {};
    for (const key of util.objectKeys(this.shape)) {
      const fieldSchema = this.shape[key];
      if (mask && !mask[key]) {
        newShape[key] = fieldSchema;
      } else {
        newShape[key] = fieldSchema.optional();
      }
    }
    return new ZodObject({
      ...this._def,
      shape: () => newShape
    });
  }
  required(mask) {
    const newShape = {};
    for (const key of util.objectKeys(this.shape)) {
      if (mask && !mask[key]) {
        newShape[key] = this.shape[key];
      } else {
        const fieldSchema = this.shape[key];
        let newField = fieldSchema;
        while (newField instanceof ZodOptional) {
          newField = newField._def.innerType;
        }
        newShape[key] = newField;
      }
    }
    return new ZodObject({
      ...this._def,
      shape: () => newShape
    });
  }
  keyof() {
    return createZodEnum(util.objectKeys(this.shape));
  }
}
ZodObject.create = (shape, params) => {
  return new ZodObject({
    shape: () => shape,
    unknownKeys: "strip",
    catchall: ZodNever.create(),
    typeName: ZodFirstPartyTypeKind.ZodObject,
    ...processCreateParams(params)
  });
};
ZodObject.strictCreate = (shape, params) => {
  return new ZodObject({
    shape: () => shape,
    unknownKeys: "strict",
    catchall: ZodNever.create(),
    typeName: ZodFirstPartyTypeKind.ZodObject,
    ...processCreateParams(params)
  });
};
ZodObject.lazycreate = (shape, params) => {
  return new ZodObject({
    shape,
    unknownKeys: "strip",
    catchall: ZodNever.create(),
    typeName: ZodFirstPartyTypeKind.ZodObject,
    ...processCreateParams(params)
  });
};

class ZodUnion extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const options = this._def.options;
    function handleResults(results) {
      for (const result of results) {
        if (result.result.status === "valid") {
          return result.result;
        }
      }
      for (const result of results) {
        if (result.result.status === "dirty") {
          ctx.common.issues.push(...result.ctx.common.issues);
          return result.result;
        }
      }
      const unionErrors = results.map((result) => new ZodError(result.ctx.common.issues));
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_union,
        unionErrors
      });
      return INVALID;
    }
    if (ctx.common.async) {
      return Promise.all(options.map(async (option) => {
        const childCtx = {
          ...ctx,
          common: {
            ...ctx.common,
            issues: []
          },
          parent: null
        };
        return {
          result: await option._parseAsync({
            data: ctx.data,
            path: ctx.path,
            parent: childCtx
          }),
          ctx: childCtx
        };
      })).then(handleResults);
    } else {
      let dirty = undefined;
      const issues = [];
      for (const option of options) {
        const childCtx = {
          ...ctx,
          common: {
            ...ctx.common,
            issues: []
          },
          parent: null
        };
        const result = option._parseSync({
          data: ctx.data,
          path: ctx.path,
          parent: childCtx
        });
        if (result.status === "valid") {
          return result;
        } else if (result.status === "dirty" && !dirty) {
          dirty = { result, ctx: childCtx };
        }
        if (childCtx.common.issues.length) {
          issues.push(childCtx.common.issues);
        }
      }
      if (dirty) {
        ctx.common.issues.push(...dirty.ctx.common.issues);
        return dirty.result;
      }
      const unionErrors = issues.map((issues2) => new ZodError(issues2));
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_union,
        unionErrors
      });
      return INVALID;
    }
  }
  get options() {
    return this._def.options;
  }
}
ZodUnion.create = (types, params) => {
  return new ZodUnion({
    options: types,
    typeName: ZodFirstPartyTypeKind.ZodUnion,
    ...processCreateParams(params)
  });
};
var getDiscriminator = (type) => {
  if (type instanceof ZodLazy) {
    return getDiscriminator(type.schema);
  } else if (type instanceof ZodEffects) {
    return getDiscriminator(type.innerType());
  } else if (type instanceof ZodLiteral) {
    return [type.value];
  } else if (type instanceof ZodEnum) {
    return type.options;
  } else if (type instanceof ZodNativeEnum) {
    return util.objectValues(type.enum);
  } else if (type instanceof ZodDefault) {
    return getDiscriminator(type._def.innerType);
  } else if (type instanceof ZodUndefined) {
    return [undefined];
  } else if (type instanceof ZodNull) {
    return [null];
  } else if (type instanceof ZodOptional) {
    return [undefined, ...getDiscriminator(type.unwrap())];
  } else if (type instanceof ZodNullable) {
    return [null, ...getDiscriminator(type.unwrap())];
  } else if (type instanceof ZodBranded) {
    return getDiscriminator(type.unwrap());
  } else if (type instanceof ZodReadonly) {
    return getDiscriminator(type.unwrap());
  } else if (type instanceof ZodCatch) {
    return getDiscriminator(type._def.innerType);
  } else {
    return [];
  }
};

class ZodDiscriminatedUnion extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.object) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.object,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const discriminator = this.discriminator;
    const discriminatorValue = ctx.data[discriminator];
    const option = this.optionsMap.get(discriminatorValue);
    if (!option) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_union_discriminator,
        options: Array.from(this.optionsMap.keys()),
        path: [discriminator]
      });
      return INVALID;
    }
    if (ctx.common.async) {
      return option._parseAsync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      });
    } else {
      return option._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      });
    }
  }
  get discriminator() {
    return this._def.discriminator;
  }
  get options() {
    return this._def.options;
  }
  get optionsMap() {
    return this._def.optionsMap;
  }
  static create(discriminator, options, params) {
    const optionsMap = new Map;
    for (const type of options) {
      const discriminatorValues = getDiscriminator(type.shape[discriminator]);
      if (!discriminatorValues.length) {
        throw new Error(`A discriminator value for key \`${discriminator}\` could not be extracted from all schema options`);
      }
      for (const value of discriminatorValues) {
        if (optionsMap.has(value)) {
          throw new Error(`Discriminator property ${String(discriminator)} has duplicate value ${String(value)}`);
        }
        optionsMap.set(value, type);
      }
    }
    return new ZodDiscriminatedUnion({
      typeName: ZodFirstPartyTypeKind.ZodDiscriminatedUnion,
      discriminator,
      options,
      optionsMap,
      ...processCreateParams(params)
    });
  }
}
function mergeValues(a, b) {
  const aType = getParsedType(a);
  const bType = getParsedType(b);
  if (a === b) {
    return { valid: true, data: a };
  } else if (aType === ZodParsedType.object && bType === ZodParsedType.object) {
    const bKeys = util.objectKeys(b);
    const sharedKeys = util.objectKeys(a).filter((key) => bKeys.indexOf(key) !== -1);
    const newObj = { ...a, ...b };
    for (const key of sharedKeys) {
      const sharedValue = mergeValues(a[key], b[key]);
      if (!sharedValue.valid) {
        return { valid: false };
      }
      newObj[key] = sharedValue.data;
    }
    return { valid: true, data: newObj };
  } else if (aType === ZodParsedType.array && bType === ZodParsedType.array) {
    if (a.length !== b.length) {
      return { valid: false };
    }
    const newArray = [];
    for (let index = 0;index < a.length; index++) {
      const itemA = a[index];
      const itemB = b[index];
      const sharedValue = mergeValues(itemA, itemB);
      if (!sharedValue.valid) {
        return { valid: false };
      }
      newArray.push(sharedValue.data);
    }
    return { valid: true, data: newArray };
  } else if (aType === ZodParsedType.date && bType === ZodParsedType.date && +a === +b) {
    return { valid: true, data: a };
  } else {
    return { valid: false };
  }
}

class ZodIntersection extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    const handleParsed = (parsedLeft, parsedRight) => {
      if (isAborted(parsedLeft) || isAborted(parsedRight)) {
        return INVALID;
      }
      const merged = mergeValues(parsedLeft.value, parsedRight.value);
      if (!merged.valid) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.invalid_intersection_types
        });
        return INVALID;
      }
      if (isDirty(parsedLeft) || isDirty(parsedRight)) {
        status.dirty();
      }
      return { status: status.value, value: merged.data };
    };
    if (ctx.common.async) {
      return Promise.all([
        this._def.left._parseAsync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        }),
        this._def.right._parseAsync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        })
      ]).then(([left, right]) => handleParsed(left, right));
    } else {
      return handleParsed(this._def.left._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      }), this._def.right._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      }));
    }
  }
}
ZodIntersection.create = (left, right, params) => {
  return new ZodIntersection({
    left,
    right,
    typeName: ZodFirstPartyTypeKind.ZodIntersection,
    ...processCreateParams(params)
  });
};

class ZodTuple extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.array) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.array,
        received: ctx.parsedType
      });
      return INVALID;
    }
    if (ctx.data.length < this._def.items.length) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.too_small,
        minimum: this._def.items.length,
        inclusive: true,
        exact: false,
        type: "array"
      });
      return INVALID;
    }
    const rest = this._def.rest;
    if (!rest && ctx.data.length > this._def.items.length) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.too_big,
        maximum: this._def.items.length,
        inclusive: true,
        exact: false,
        type: "array"
      });
      status.dirty();
    }
    const items = [...ctx.data].map((item, itemIndex) => {
      const schema = this._def.items[itemIndex] || this._def.rest;
      if (!schema)
        return null;
      return schema._parse(new ParseInputLazyPath(ctx, item, ctx.path, itemIndex));
    }).filter((x) => !!x);
    if (ctx.common.async) {
      return Promise.all(items).then((results) => {
        return ParseStatus.mergeArray(status, results);
      });
    } else {
      return ParseStatus.mergeArray(status, items);
    }
  }
  get items() {
    return this._def.items;
  }
  rest(rest) {
    return new ZodTuple({
      ...this._def,
      rest
    });
  }
}
ZodTuple.create = (schemas, params) => {
  if (!Array.isArray(schemas)) {
    throw new Error("You must pass an array of schemas to z.tuple([ ... ])");
  }
  return new ZodTuple({
    items: schemas,
    typeName: ZodFirstPartyTypeKind.ZodTuple,
    rest: null,
    ...processCreateParams(params)
  });
};

class ZodRecord extends ZodType {
  get keySchema() {
    return this._def.keyType;
  }
  get valueSchema() {
    return this._def.valueType;
  }
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.object) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.object,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const pairs = [];
    const keyType = this._def.keyType;
    const valueType = this._def.valueType;
    for (const key in ctx.data) {
      pairs.push({
        key: keyType._parse(new ParseInputLazyPath(ctx, key, ctx.path, key)),
        value: valueType._parse(new ParseInputLazyPath(ctx, ctx.data[key], ctx.path, key)),
        alwaysSet: key in ctx.data
      });
    }
    if (ctx.common.async) {
      return ParseStatus.mergeObjectAsync(status, pairs);
    } else {
      return ParseStatus.mergeObjectSync(status, pairs);
    }
  }
  get element() {
    return this._def.valueType;
  }
  static create(first, second, third) {
    if (second instanceof ZodType) {
      return new ZodRecord({
        keyType: first,
        valueType: second,
        typeName: ZodFirstPartyTypeKind.ZodRecord,
        ...processCreateParams(third)
      });
    }
    return new ZodRecord({
      keyType: ZodString.create(),
      valueType: first,
      typeName: ZodFirstPartyTypeKind.ZodRecord,
      ...processCreateParams(second)
    });
  }
}

class ZodMap extends ZodType {
  get keySchema() {
    return this._def.keyType;
  }
  get valueSchema() {
    return this._def.valueType;
  }
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.map) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.map,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const keyType = this._def.keyType;
    const valueType = this._def.valueType;
    const pairs = [...ctx.data.entries()].map(([key, value], index) => {
      return {
        key: keyType._parse(new ParseInputLazyPath(ctx, key, ctx.path, [index, "key"])),
        value: valueType._parse(new ParseInputLazyPath(ctx, value, ctx.path, [index, "value"]))
      };
    });
    if (ctx.common.async) {
      const finalMap = new Map;
      return Promise.resolve().then(async () => {
        for (const pair of pairs) {
          const key = await pair.key;
          const value = await pair.value;
          if (key.status === "aborted" || value.status === "aborted") {
            return INVALID;
          }
          if (key.status === "dirty" || value.status === "dirty") {
            status.dirty();
          }
          finalMap.set(key.value, value.value);
        }
        return { status: status.value, value: finalMap };
      });
    } else {
      const finalMap = new Map;
      for (const pair of pairs) {
        const key = pair.key;
        const value = pair.value;
        if (key.status === "aborted" || value.status === "aborted") {
          return INVALID;
        }
        if (key.status === "dirty" || value.status === "dirty") {
          status.dirty();
        }
        finalMap.set(key.value, value.value);
      }
      return { status: status.value, value: finalMap };
    }
  }
}
ZodMap.create = (keyType, valueType, params) => {
  return new ZodMap({
    valueType,
    keyType,
    typeName: ZodFirstPartyTypeKind.ZodMap,
    ...processCreateParams(params)
  });
};

class ZodSet extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.set) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.set,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const def = this._def;
    if (def.minSize !== null) {
      if (ctx.data.size < def.minSize.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_small,
          minimum: def.minSize.value,
          type: "set",
          inclusive: true,
          exact: false,
          message: def.minSize.message
        });
        status.dirty();
      }
    }
    if (def.maxSize !== null) {
      if (ctx.data.size > def.maxSize.value) {
        addIssueToContext(ctx, {
          code: ZodIssueCode.too_big,
          maximum: def.maxSize.value,
          type: "set",
          inclusive: true,
          exact: false,
          message: def.maxSize.message
        });
        status.dirty();
      }
    }
    const valueType = this._def.valueType;
    function finalizeSet(elements2) {
      const parsedSet = new Set;
      for (const element of elements2) {
        if (element.status === "aborted")
          return INVALID;
        if (element.status === "dirty")
          status.dirty();
        parsedSet.add(element.value);
      }
      return { status: status.value, value: parsedSet };
    }
    const elements = [...ctx.data.values()].map((item, i) => valueType._parse(new ParseInputLazyPath(ctx, item, ctx.path, i)));
    if (ctx.common.async) {
      return Promise.all(elements).then((elements2) => finalizeSet(elements2));
    } else {
      return finalizeSet(elements);
    }
  }
  min(minSize, message) {
    return new ZodSet({
      ...this._def,
      minSize: { value: minSize, message: errorUtil.toString(message) }
    });
  }
  max(maxSize, message) {
    return new ZodSet({
      ...this._def,
      maxSize: { value: maxSize, message: errorUtil.toString(message) }
    });
  }
  size(size, message) {
    return this.min(size, message).max(size, message);
  }
  nonempty(message) {
    return this.min(1, message);
  }
}
ZodSet.create = (valueType, params) => {
  return new ZodSet({
    valueType,
    minSize: null,
    maxSize: null,
    typeName: ZodFirstPartyTypeKind.ZodSet,
    ...processCreateParams(params)
  });
};

class ZodFunction extends ZodType {
  constructor() {
    super(...arguments);
    this.validate = this.implement;
  }
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.function) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.function,
        received: ctx.parsedType
      });
      return INVALID;
    }
    function makeArgsIssue(args, error) {
      return makeIssue({
        data: args,
        path: ctx.path,
        errorMaps: [ctx.common.contextualErrorMap, ctx.schemaErrorMap, getErrorMap(), en_default].filter((x) => !!x),
        issueData: {
          code: ZodIssueCode.invalid_arguments,
          argumentsError: error
        }
      });
    }
    function makeReturnsIssue(returns, error) {
      return makeIssue({
        data: returns,
        path: ctx.path,
        errorMaps: [ctx.common.contextualErrorMap, ctx.schemaErrorMap, getErrorMap(), en_default].filter((x) => !!x),
        issueData: {
          code: ZodIssueCode.invalid_return_type,
          returnTypeError: error
        }
      });
    }
    const params = { errorMap: ctx.common.contextualErrorMap };
    const fn = ctx.data;
    if (this._def.returns instanceof ZodPromise) {
      const me = this;
      return OK(async function(...args) {
        const error = new ZodError([]);
        const parsedArgs = await me._def.args.parseAsync(args, params).catch((e) => {
          error.addIssue(makeArgsIssue(args, e));
          throw error;
        });
        const result = await Reflect.apply(fn, this, parsedArgs);
        const parsedReturns = await me._def.returns._def.type.parseAsync(result, params).catch((e) => {
          error.addIssue(makeReturnsIssue(result, e));
          throw error;
        });
        return parsedReturns;
      });
    } else {
      const me = this;
      return OK(function(...args) {
        const parsedArgs = me._def.args.safeParse(args, params);
        if (!parsedArgs.success) {
          throw new ZodError([makeArgsIssue(args, parsedArgs.error)]);
        }
        const result = Reflect.apply(fn, this, parsedArgs.data);
        const parsedReturns = me._def.returns.safeParse(result, params);
        if (!parsedReturns.success) {
          throw new ZodError([makeReturnsIssue(result, parsedReturns.error)]);
        }
        return parsedReturns.data;
      });
    }
  }
  parameters() {
    return this._def.args;
  }
  returnType() {
    return this._def.returns;
  }
  args(...items) {
    return new ZodFunction({
      ...this._def,
      args: ZodTuple.create(items).rest(ZodUnknown.create())
    });
  }
  returns(returnType) {
    return new ZodFunction({
      ...this._def,
      returns: returnType
    });
  }
  implement(func) {
    const validatedFunc = this.parse(func);
    return validatedFunc;
  }
  strictImplement(func) {
    const validatedFunc = this.parse(func);
    return validatedFunc;
  }
  static create(args, returns, params) {
    return new ZodFunction({
      args: args ? args : ZodTuple.create([]).rest(ZodUnknown.create()),
      returns: returns || ZodUnknown.create(),
      typeName: ZodFirstPartyTypeKind.ZodFunction,
      ...processCreateParams(params)
    });
  }
}

class ZodLazy extends ZodType {
  get schema() {
    return this._def.getter();
  }
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const lazySchema = this._def.getter();
    return lazySchema._parse({ data: ctx.data, path: ctx.path, parent: ctx });
  }
}
ZodLazy.create = (getter, params) => {
  return new ZodLazy({
    getter,
    typeName: ZodFirstPartyTypeKind.ZodLazy,
    ...processCreateParams(params)
  });
};

class ZodLiteral extends ZodType {
  _parse(input) {
    if (input.data !== this._def.value) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        received: ctx.data,
        code: ZodIssueCode.invalid_literal,
        expected: this._def.value
      });
      return INVALID;
    }
    return { status: "valid", value: input.data };
  }
  get value() {
    return this._def.value;
  }
}
ZodLiteral.create = (value, params) => {
  return new ZodLiteral({
    value,
    typeName: ZodFirstPartyTypeKind.ZodLiteral,
    ...processCreateParams(params)
  });
};
function createZodEnum(values, params) {
  return new ZodEnum({
    values,
    typeName: ZodFirstPartyTypeKind.ZodEnum,
    ...processCreateParams(params)
  });
}

class ZodEnum extends ZodType {
  _parse(input) {
    if (typeof input.data !== "string") {
      const ctx = this._getOrReturnCtx(input);
      const expectedValues = this._def.values;
      addIssueToContext(ctx, {
        expected: util.joinValues(expectedValues),
        received: ctx.parsedType,
        code: ZodIssueCode.invalid_type
      });
      return INVALID;
    }
    if (!this._cache) {
      this._cache = new Set(this._def.values);
    }
    if (!this._cache.has(input.data)) {
      const ctx = this._getOrReturnCtx(input);
      const expectedValues = this._def.values;
      addIssueToContext(ctx, {
        received: ctx.data,
        code: ZodIssueCode.invalid_enum_value,
        options: expectedValues
      });
      return INVALID;
    }
    return OK(input.data);
  }
  get options() {
    return this._def.values;
  }
  get enum() {
    const enumValues = {};
    for (const val of this._def.values) {
      enumValues[val] = val;
    }
    return enumValues;
  }
  get Values() {
    const enumValues = {};
    for (const val of this._def.values) {
      enumValues[val] = val;
    }
    return enumValues;
  }
  get Enum() {
    const enumValues = {};
    for (const val of this._def.values) {
      enumValues[val] = val;
    }
    return enumValues;
  }
  extract(values, newDef = this._def) {
    return ZodEnum.create(values, {
      ...this._def,
      ...newDef
    });
  }
  exclude(values, newDef = this._def) {
    return ZodEnum.create(this.options.filter((opt) => !values.includes(opt)), {
      ...this._def,
      ...newDef
    });
  }
}
ZodEnum.create = createZodEnum;

class ZodNativeEnum extends ZodType {
  _parse(input) {
    const nativeEnumValues = util.getValidEnumValues(this._def.values);
    const ctx = this._getOrReturnCtx(input);
    if (ctx.parsedType !== ZodParsedType.string && ctx.parsedType !== ZodParsedType.number) {
      const expectedValues = util.objectValues(nativeEnumValues);
      addIssueToContext(ctx, {
        expected: util.joinValues(expectedValues),
        received: ctx.parsedType,
        code: ZodIssueCode.invalid_type
      });
      return INVALID;
    }
    if (!this._cache) {
      this._cache = new Set(util.getValidEnumValues(this._def.values));
    }
    if (!this._cache.has(input.data)) {
      const expectedValues = util.objectValues(nativeEnumValues);
      addIssueToContext(ctx, {
        received: ctx.data,
        code: ZodIssueCode.invalid_enum_value,
        options: expectedValues
      });
      return INVALID;
    }
    return OK(input.data);
  }
  get enum() {
    return this._def.values;
  }
}
ZodNativeEnum.create = (values, params) => {
  return new ZodNativeEnum({
    values,
    typeName: ZodFirstPartyTypeKind.ZodNativeEnum,
    ...processCreateParams(params)
  });
};

class ZodPromise extends ZodType {
  unwrap() {
    return this._def.type;
  }
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    if (ctx.parsedType !== ZodParsedType.promise && ctx.common.async === false) {
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.promise,
        received: ctx.parsedType
      });
      return INVALID;
    }
    const promisified = ctx.parsedType === ZodParsedType.promise ? ctx.data : Promise.resolve(ctx.data);
    return OK(promisified.then((data) => {
      return this._def.type.parseAsync(data, {
        path: ctx.path,
        errorMap: ctx.common.contextualErrorMap
      });
    }));
  }
}
ZodPromise.create = (schema, params) => {
  return new ZodPromise({
    type: schema,
    typeName: ZodFirstPartyTypeKind.ZodPromise,
    ...processCreateParams(params)
  });
};

class ZodEffects extends ZodType {
  innerType() {
    return this._def.schema;
  }
  sourceType() {
    return this._def.schema._def.typeName === ZodFirstPartyTypeKind.ZodEffects ? this._def.schema.sourceType() : this._def.schema;
  }
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    const effect = this._def.effect || null;
    const checkCtx = {
      addIssue: (arg) => {
        addIssueToContext(ctx, arg);
        if (arg.fatal) {
          status.abort();
        } else {
          status.dirty();
        }
      },
      get path() {
        return ctx.path;
      }
    };
    checkCtx.addIssue = checkCtx.addIssue.bind(checkCtx);
    if (effect.type === "preprocess") {
      const processed = effect.transform(ctx.data, checkCtx);
      if (ctx.common.async) {
        return Promise.resolve(processed).then(async (processed2) => {
          if (status.value === "aborted")
            return INVALID;
          const result = await this._def.schema._parseAsync({
            data: processed2,
            path: ctx.path,
            parent: ctx
          });
          if (result.status === "aborted")
            return INVALID;
          if (result.status === "dirty")
            return DIRTY(result.value);
          if (status.value === "dirty")
            return DIRTY(result.value);
          return result;
        });
      } else {
        if (status.value === "aborted")
          return INVALID;
        const result = this._def.schema._parseSync({
          data: processed,
          path: ctx.path,
          parent: ctx
        });
        if (result.status === "aborted")
          return INVALID;
        if (result.status === "dirty")
          return DIRTY(result.value);
        if (status.value === "dirty")
          return DIRTY(result.value);
        return result;
      }
    }
    if (effect.type === "refinement") {
      const executeRefinement = (acc) => {
        const result = effect.refinement(acc, checkCtx);
        if (ctx.common.async) {
          return Promise.resolve(result);
        }
        if (result instanceof Promise) {
          throw new Error("Async refinement encountered during synchronous parse operation. Use .parseAsync instead.");
        }
        return acc;
      };
      if (ctx.common.async === false) {
        const inner = this._def.schema._parseSync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        });
        if (inner.status === "aborted")
          return INVALID;
        if (inner.status === "dirty")
          status.dirty();
        executeRefinement(inner.value);
        return { status: status.value, value: inner.value };
      } else {
        return this._def.schema._parseAsync({ data: ctx.data, path: ctx.path, parent: ctx }).then((inner) => {
          if (inner.status === "aborted")
            return INVALID;
          if (inner.status === "dirty")
            status.dirty();
          return executeRefinement(inner.value).then(() => {
            return { status: status.value, value: inner.value };
          });
        });
      }
    }
    if (effect.type === "transform") {
      if (ctx.common.async === false) {
        const base = this._def.schema._parseSync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        });
        if (!isValid(base))
          return INVALID;
        const result = effect.transform(base.value, checkCtx);
        if (result instanceof Promise) {
          throw new Error(`Asynchronous transform encountered during synchronous parse operation. Use .parseAsync instead.`);
        }
        return { status: status.value, value: result };
      } else {
        return this._def.schema._parseAsync({ data: ctx.data, path: ctx.path, parent: ctx }).then((base) => {
          if (!isValid(base))
            return INVALID;
          return Promise.resolve(effect.transform(base.value, checkCtx)).then((result) => ({
            status: status.value,
            value: result
          }));
        });
      }
    }
    util.assertNever(effect);
  }
}
ZodEffects.create = (schema, effect, params) => {
  return new ZodEffects({
    schema,
    typeName: ZodFirstPartyTypeKind.ZodEffects,
    effect,
    ...processCreateParams(params)
  });
};
ZodEffects.createWithPreprocess = (preprocess, schema, params) => {
  return new ZodEffects({
    schema,
    effect: { type: "preprocess", transform: preprocess },
    typeName: ZodFirstPartyTypeKind.ZodEffects,
    ...processCreateParams(params)
  });
};
class ZodOptional extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType === ZodParsedType.undefined) {
      return OK(undefined);
    }
    return this._def.innerType._parse(input);
  }
  unwrap() {
    return this._def.innerType;
  }
}
ZodOptional.create = (type, params) => {
  return new ZodOptional({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodOptional,
    ...processCreateParams(params)
  });
};

class ZodNullable extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType === ZodParsedType.null) {
      return OK(null);
    }
    return this._def.innerType._parse(input);
  }
  unwrap() {
    return this._def.innerType;
  }
}
ZodNullable.create = (type, params) => {
  return new ZodNullable({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodNullable,
    ...processCreateParams(params)
  });
};

class ZodDefault extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    let data = ctx.data;
    if (ctx.parsedType === ZodParsedType.undefined) {
      data = this._def.defaultValue();
    }
    return this._def.innerType._parse({
      data,
      path: ctx.path,
      parent: ctx
    });
  }
  removeDefault() {
    return this._def.innerType;
  }
}
ZodDefault.create = (type, params) => {
  return new ZodDefault({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodDefault,
    defaultValue: typeof params.default === "function" ? params.default : () => params.default,
    ...processCreateParams(params)
  });
};

class ZodCatch extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const newCtx = {
      ...ctx,
      common: {
        ...ctx.common,
        issues: []
      }
    };
    const result = this._def.innerType._parse({
      data: newCtx.data,
      path: newCtx.path,
      parent: {
        ...newCtx
      }
    });
    if (isAsync(result)) {
      return result.then((result2) => {
        return {
          status: "valid",
          value: result2.status === "valid" ? result2.value : this._def.catchValue({
            get error() {
              return new ZodError(newCtx.common.issues);
            },
            input: newCtx.data
          })
        };
      });
    } else {
      return {
        status: "valid",
        value: result.status === "valid" ? result.value : this._def.catchValue({
          get error() {
            return new ZodError(newCtx.common.issues);
          },
          input: newCtx.data
        })
      };
    }
  }
  removeCatch() {
    return this._def.innerType;
  }
}
ZodCatch.create = (type, params) => {
  return new ZodCatch({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodCatch,
    catchValue: typeof params.catch === "function" ? params.catch : () => params.catch,
    ...processCreateParams(params)
  });
};

class ZodNaN extends ZodType {
  _parse(input) {
    const parsedType = this._getType(input);
    if (parsedType !== ZodParsedType.nan) {
      const ctx = this._getOrReturnCtx(input);
      addIssueToContext(ctx, {
        code: ZodIssueCode.invalid_type,
        expected: ZodParsedType.nan,
        received: ctx.parsedType
      });
      return INVALID;
    }
    return { status: "valid", value: input.data };
  }
}
ZodNaN.create = (params) => {
  return new ZodNaN({
    typeName: ZodFirstPartyTypeKind.ZodNaN,
    ...processCreateParams(params)
  });
};
var BRAND = Symbol("zod_brand");

class ZodBranded extends ZodType {
  _parse(input) {
    const { ctx } = this._processInputParams(input);
    const data = ctx.data;
    return this._def.type._parse({
      data,
      path: ctx.path,
      parent: ctx
    });
  }
  unwrap() {
    return this._def.type;
  }
}

class ZodPipeline extends ZodType {
  _parse(input) {
    const { status, ctx } = this._processInputParams(input);
    if (ctx.common.async) {
      const handleAsync = async () => {
        const inResult = await this._def.in._parseAsync({
          data: ctx.data,
          path: ctx.path,
          parent: ctx
        });
        if (inResult.status === "aborted")
          return INVALID;
        if (inResult.status === "dirty") {
          status.dirty();
          return DIRTY(inResult.value);
        } else {
          return this._def.out._parseAsync({
            data: inResult.value,
            path: ctx.path,
            parent: ctx
          });
        }
      };
      return handleAsync();
    } else {
      const inResult = this._def.in._parseSync({
        data: ctx.data,
        path: ctx.path,
        parent: ctx
      });
      if (inResult.status === "aborted")
        return INVALID;
      if (inResult.status === "dirty") {
        status.dirty();
        return {
          status: "dirty",
          value: inResult.value
        };
      } else {
        return this._def.out._parseSync({
          data: inResult.value,
          path: ctx.path,
          parent: ctx
        });
      }
    }
  }
  static create(a, b) {
    return new ZodPipeline({
      in: a,
      out: b,
      typeName: ZodFirstPartyTypeKind.ZodPipeline
    });
  }
}

class ZodReadonly extends ZodType {
  _parse(input) {
    const result = this._def.innerType._parse(input);
    const freeze = (data) => {
      if (isValid(data)) {
        data.value = Object.freeze(data.value);
      }
      return data;
    };
    return isAsync(result) ? result.then((data) => freeze(data)) : freeze(result);
  }
  unwrap() {
    return this._def.innerType;
  }
}
ZodReadonly.create = (type, params) => {
  return new ZodReadonly({
    innerType: type,
    typeName: ZodFirstPartyTypeKind.ZodReadonly,
    ...processCreateParams(params)
  });
};
function cleanParams(params, data) {
  const p = typeof params === "function" ? params(data) : typeof params === "string" ? { message: params } : params;
  const p2 = typeof p === "string" ? { message: p } : p;
  return p2;
}
function custom(check, _params = {}, fatal) {
  if (check)
    return ZodAny.create().superRefine((data, ctx) => {
      const r = check(data);
      if (r instanceof Promise) {
        return r.then((r2) => {
          if (!r2) {
            const params = cleanParams(_params, data);
            const _fatal = params.fatal ?? fatal ?? true;
            ctx.addIssue({ code: "custom", ...params, fatal: _fatal });
          }
        });
      }
      if (!r) {
        const params = cleanParams(_params, data);
        const _fatal = params.fatal ?? fatal ?? true;
        ctx.addIssue({ code: "custom", ...params, fatal: _fatal });
      }
      return;
    });
  return ZodAny.create();
}
var late = {
  object: ZodObject.lazycreate
};
var ZodFirstPartyTypeKind;
(function(ZodFirstPartyTypeKind2) {
  ZodFirstPartyTypeKind2["ZodString"] = "ZodString";
  ZodFirstPartyTypeKind2["ZodNumber"] = "ZodNumber";
  ZodFirstPartyTypeKind2["ZodNaN"] = "ZodNaN";
  ZodFirstPartyTypeKind2["ZodBigInt"] = "ZodBigInt";
  ZodFirstPartyTypeKind2["ZodBoolean"] = "ZodBoolean";
  ZodFirstPartyTypeKind2["ZodDate"] = "ZodDate";
  ZodFirstPartyTypeKind2["ZodSymbol"] = "ZodSymbol";
  ZodFirstPartyTypeKind2["ZodUndefined"] = "ZodUndefined";
  ZodFirstPartyTypeKind2["ZodNull"] = "ZodNull";
  ZodFirstPartyTypeKind2["ZodAny"] = "ZodAny";
  ZodFirstPartyTypeKind2["ZodUnknown"] = "ZodUnknown";
  ZodFirstPartyTypeKind2["ZodNever"] = "ZodNever";
  ZodFirstPartyTypeKind2["ZodVoid"] = "ZodVoid";
  ZodFirstPartyTypeKind2["ZodArray"] = "ZodArray";
  ZodFirstPartyTypeKind2["ZodObject"] = "ZodObject";
  ZodFirstPartyTypeKind2["ZodUnion"] = "ZodUnion";
  ZodFirstPartyTypeKind2["ZodDiscriminatedUnion"] = "ZodDiscriminatedUnion";
  ZodFirstPartyTypeKind2["ZodIntersection"] = "ZodIntersection";
  ZodFirstPartyTypeKind2["ZodTuple"] = "ZodTuple";
  ZodFirstPartyTypeKind2["ZodRecord"] = "ZodRecord";
  ZodFirstPartyTypeKind2["ZodMap"] = "ZodMap";
  ZodFirstPartyTypeKind2["ZodSet"] = "ZodSet";
  ZodFirstPartyTypeKind2["ZodFunction"] = "ZodFunction";
  ZodFirstPartyTypeKind2["ZodLazy"] = "ZodLazy";
  ZodFirstPartyTypeKind2["ZodLiteral"] = "ZodLiteral";
  ZodFirstPartyTypeKind2["ZodEnum"] = "ZodEnum";
  ZodFirstPartyTypeKind2["ZodEffects"] = "ZodEffects";
  ZodFirstPartyTypeKind2["ZodNativeEnum"] = "ZodNativeEnum";
  ZodFirstPartyTypeKind2["ZodOptional"] = "ZodOptional";
  ZodFirstPartyTypeKind2["ZodNullable"] = "ZodNullable";
  ZodFirstPartyTypeKind2["ZodDefault"] = "ZodDefault";
  ZodFirstPartyTypeKind2["ZodCatch"] = "ZodCatch";
  ZodFirstPartyTypeKind2["ZodPromise"] = "ZodPromise";
  ZodFirstPartyTypeKind2["ZodBranded"] = "ZodBranded";
  ZodFirstPartyTypeKind2["ZodPipeline"] = "ZodPipeline";
  ZodFirstPartyTypeKind2["ZodReadonly"] = "ZodReadonly";
})(ZodFirstPartyTypeKind || (ZodFirstPartyTypeKind = {}));
var instanceOfType = (cls, params = {
  message: `Input not instance of ${cls.name}`
}) => custom((data) => data instanceof cls, params);
var stringType = ZodString.create;
var numberType = ZodNumber.create;
var nanType = ZodNaN.create;
var bigIntType = ZodBigInt.create;
var booleanType = ZodBoolean.create;
var dateType = ZodDate.create;
var symbolType = ZodSymbol.create;
var undefinedType = ZodUndefined.create;
var nullType = ZodNull.create;
var anyType = ZodAny.create;
var unknownType = ZodUnknown.create;
var neverType = ZodNever.create;
var voidType = ZodVoid.create;
var arrayType = ZodArray.create;
var objectType = ZodObject.create;
var strictObjectType = ZodObject.strictCreate;
var unionType = ZodUnion.create;
var discriminatedUnionType = ZodDiscriminatedUnion.create;
var intersectionType = ZodIntersection.create;
var tupleType = ZodTuple.create;
var recordType = ZodRecord.create;
var mapType = ZodMap.create;
var setType = ZodSet.create;
var functionType = ZodFunction.create;
var lazyType = ZodLazy.create;
var literalType = ZodLiteral.create;
var enumType = ZodEnum.create;
var nativeEnumType = ZodNativeEnum.create;
var promiseType = ZodPromise.create;
var effectsType = ZodEffects.create;
var optionalType = ZodOptional.create;
var nullableType = ZodNullable.create;
var preprocessType = ZodEffects.createWithPreprocess;
var pipelineType = ZodPipeline.create;
var ostring = () => stringType().optional();
var onumber = () => numberType().optional();
var oboolean = () => booleanType().optional();
var coerce = {
  string: (arg) => ZodString.create({ ...arg, coerce: true }),
  number: (arg) => ZodNumber.create({ ...arg, coerce: true }),
  boolean: (arg) => ZodBoolean.create({
    ...arg,
    coerce: true
  }),
  bigint: (arg) => ZodBigInt.create({ ...arg, coerce: true }),
  date: (arg) => ZodDate.create({ ...arg, coerce: true })
};
var NEVER = INVALID;
// node_modules/@modelcontextprotocol/sdk/dist/types.js
var LATEST_PROTOCOL_VERSION = "2024-11-05";
var SUPPORTED_PROTOCOL_VERSIONS = [
  LATEST_PROTOCOL_VERSION,
  "2024-10-07"
];
var JSONRPC_VERSION = "2.0";
var ProgressTokenSchema = exports_external.union([exports_external.string(), exports_external.number().int()]);
var CursorSchema = exports_external.string();
var BaseRequestParamsSchema = exports_external.object({
  _meta: exports_external.optional(exports_external.object({
    progressToken: exports_external.optional(ProgressTokenSchema)
  }).passthrough())
}).passthrough();
var RequestSchema = exports_external.object({
  method: exports_external.string(),
  params: exports_external.optional(BaseRequestParamsSchema)
});
var BaseNotificationParamsSchema = exports_external.object({
  _meta: exports_external.optional(exports_external.object({}).passthrough())
}).passthrough();
var NotificationSchema = exports_external.object({
  method: exports_external.string(),
  params: exports_external.optional(BaseNotificationParamsSchema)
});
var ResultSchema = exports_external.object({
  _meta: exports_external.optional(exports_external.object({}).passthrough())
}).passthrough();
var RequestIdSchema = exports_external.union([exports_external.string(), exports_external.number().int()]);
var JSONRPCRequestSchema = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION),
  id: RequestIdSchema
}).merge(RequestSchema).strict();
var JSONRPCNotificationSchema = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION)
}).merge(NotificationSchema).strict();
var JSONRPCResponseSchema = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION),
  id: RequestIdSchema,
  result: ResultSchema
}).strict();
var ErrorCode;
(function(ErrorCode2) {
  ErrorCode2[ErrorCode2["ConnectionClosed"] = -1] = "ConnectionClosed";
  ErrorCode2[ErrorCode2["RequestTimeout"] = -2] = "RequestTimeout";
  ErrorCode2[ErrorCode2["ParseError"] = -32700] = "ParseError";
  ErrorCode2[ErrorCode2["InvalidRequest"] = -32600] = "InvalidRequest";
  ErrorCode2[ErrorCode2["MethodNotFound"] = -32601] = "MethodNotFound";
  ErrorCode2[ErrorCode2["InvalidParams"] = -32602] = "InvalidParams";
  ErrorCode2[ErrorCode2["InternalError"] = -32603] = "InternalError";
})(ErrorCode || (ErrorCode = {}));
var JSONRPCErrorSchema = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION),
  id: RequestIdSchema,
  error: exports_external.object({
    code: exports_external.number().int(),
    message: exports_external.string(),
    data: exports_external.optional(exports_external.unknown())
  })
}).strict();
var JSONRPCMessageSchema = exports_external.union([
  JSONRPCRequestSchema,
  JSONRPCNotificationSchema,
  JSONRPCResponseSchema,
  JSONRPCErrorSchema
]);
var EmptyResultSchema = ResultSchema.strict();
var CancelledNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/cancelled"),
  params: BaseNotificationParamsSchema.extend({
    requestId: RequestIdSchema,
    reason: exports_external.string().optional()
  })
});
var ImplementationSchema = exports_external.object({
  name: exports_external.string(),
  version: exports_external.string()
}).passthrough();
var ClientCapabilitiesSchema = exports_external.object({
  experimental: exports_external.optional(exports_external.object({}).passthrough()),
  sampling: exports_external.optional(exports_external.object({}).passthrough()),
  roots: exports_external.optional(exports_external.object({
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough())
}).passthrough();
var InitializeRequestSchema = RequestSchema.extend({
  method: exports_external.literal("initialize"),
  params: BaseRequestParamsSchema.extend({
    protocolVersion: exports_external.string(),
    capabilities: ClientCapabilitiesSchema,
    clientInfo: ImplementationSchema
  })
});
var ServerCapabilitiesSchema = exports_external.object({
  experimental: exports_external.optional(exports_external.object({}).passthrough()),
  logging: exports_external.optional(exports_external.object({}).passthrough()),
  prompts: exports_external.optional(exports_external.object({
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough()),
  resources: exports_external.optional(exports_external.object({
    subscribe: exports_external.optional(exports_external.boolean()),
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough()),
  tools: exports_external.optional(exports_external.object({
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough())
}).passthrough();
var InitializeResultSchema = ResultSchema.extend({
  protocolVersion: exports_external.string(),
  capabilities: ServerCapabilitiesSchema,
  serverInfo: ImplementationSchema
});
var InitializedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/initialized")
});
var PingRequestSchema = RequestSchema.extend({
  method: exports_external.literal("ping")
});
var ProgressSchema = exports_external.object({
  progress: exports_external.number(),
  total: exports_external.optional(exports_external.number())
}).passthrough();
var ProgressNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/progress"),
  params: BaseNotificationParamsSchema.merge(ProgressSchema).extend({
    progressToken: ProgressTokenSchema
  })
});
var PaginatedRequestSchema = RequestSchema.extend({
  params: BaseRequestParamsSchema.extend({
    cursor: exports_external.optional(CursorSchema)
  }).optional()
});
var PaginatedResultSchema = ResultSchema.extend({
  nextCursor: exports_external.optional(CursorSchema)
});
var ResourceContentsSchema = exports_external.object({
  uri: exports_external.string(),
  mimeType: exports_external.optional(exports_external.string())
}).passthrough();
var TextResourceContentsSchema = ResourceContentsSchema.extend({
  text: exports_external.string()
});
var BlobResourceContentsSchema = ResourceContentsSchema.extend({
  blob: exports_external.string().base64()
});
var ResourceSchema = exports_external.object({
  uri: exports_external.string(),
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  mimeType: exports_external.optional(exports_external.string())
}).passthrough();
var ResourceTemplateSchema = exports_external.object({
  uriTemplate: exports_external.string(),
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  mimeType: exports_external.optional(exports_external.string())
}).passthrough();
var ListResourcesRequestSchema = PaginatedRequestSchema.extend({
  method: exports_external.literal("resources/list")
});
var ListResourcesResultSchema = PaginatedResultSchema.extend({
  resources: exports_external.array(ResourceSchema)
});
var ListResourceTemplatesRequestSchema = PaginatedRequestSchema.extend({
  method: exports_external.literal("resources/templates/list")
});
var ListResourceTemplatesResultSchema = PaginatedResultSchema.extend({
  resourceTemplates: exports_external.array(ResourceTemplateSchema)
});
var ReadResourceRequestSchema = RequestSchema.extend({
  method: exports_external.literal("resources/read"),
  params: BaseRequestParamsSchema.extend({
    uri: exports_external.string()
  })
});
var ReadResourceResultSchema = ResultSchema.extend({
  contents: exports_external.array(exports_external.union([TextResourceContentsSchema, BlobResourceContentsSchema]))
});
var ResourceListChangedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/resources/list_changed")
});
var SubscribeRequestSchema = RequestSchema.extend({
  method: exports_external.literal("resources/subscribe"),
  params: BaseRequestParamsSchema.extend({
    uri: exports_external.string()
  })
});
var UnsubscribeRequestSchema = RequestSchema.extend({
  method: exports_external.literal("resources/unsubscribe"),
  params: BaseRequestParamsSchema.extend({
    uri: exports_external.string()
  })
});
var ResourceUpdatedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/resources/updated"),
  params: BaseNotificationParamsSchema.extend({
    uri: exports_external.string()
  })
});
var PromptArgumentSchema = exports_external.object({
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  required: exports_external.optional(exports_external.boolean())
}).passthrough();
var PromptSchema = exports_external.object({
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  arguments: exports_external.optional(exports_external.array(PromptArgumentSchema))
}).passthrough();
var ListPromptsRequestSchema = PaginatedRequestSchema.extend({
  method: exports_external.literal("prompts/list")
});
var ListPromptsResultSchema = PaginatedResultSchema.extend({
  prompts: exports_external.array(PromptSchema)
});
var GetPromptRequestSchema = RequestSchema.extend({
  method: exports_external.literal("prompts/get"),
  params: BaseRequestParamsSchema.extend({
    name: exports_external.string(),
    arguments: exports_external.optional(exports_external.record(exports_external.string()))
  })
});
var TextContentSchema = exports_external.object({
  type: exports_external.literal("text"),
  text: exports_external.string()
}).passthrough();
var ImageContentSchema = exports_external.object({
  type: exports_external.literal("image"),
  data: exports_external.string().base64(),
  mimeType: exports_external.string()
}).passthrough();
var EmbeddedResourceSchema = exports_external.object({
  type: exports_external.literal("resource"),
  resource: exports_external.union([TextResourceContentsSchema, BlobResourceContentsSchema])
}).passthrough();
var PromptMessageSchema = exports_external.object({
  role: exports_external.enum(["user", "assistant"]),
  content: exports_external.union([
    TextContentSchema,
    ImageContentSchema,
    EmbeddedResourceSchema
  ])
}).passthrough();
var GetPromptResultSchema = ResultSchema.extend({
  description: exports_external.optional(exports_external.string()),
  messages: exports_external.array(PromptMessageSchema)
});
var PromptListChangedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/prompts/list_changed")
});
var ToolSchema = exports_external.object({
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  inputSchema: exports_external.object({
    type: exports_external.literal("object"),
    properties: exports_external.optional(exports_external.object({}).passthrough())
  }).passthrough()
}).passthrough();
var ListToolsRequestSchema = PaginatedRequestSchema.extend({
  method: exports_external.literal("tools/list")
});
var ListToolsResultSchema = PaginatedResultSchema.extend({
  tools: exports_external.array(ToolSchema)
});
var CallToolResultSchema = ResultSchema.extend({
  content: exports_external.array(exports_external.union([TextContentSchema, ImageContentSchema, EmbeddedResourceSchema])),
  isError: exports_external.boolean().default(false).optional()
});
var CompatibilityCallToolResultSchema = CallToolResultSchema.or(ResultSchema.extend({
  toolResult: exports_external.unknown()
}));
var CallToolRequestSchema = RequestSchema.extend({
  method: exports_external.literal("tools/call"),
  params: BaseRequestParamsSchema.extend({
    name: exports_external.string(),
    arguments: exports_external.optional(exports_external.record(exports_external.unknown()))
  })
});
var ToolListChangedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/tools/list_changed")
});
var LoggingLevelSchema = exports_external.enum([
  "debug",
  "info",
  "notice",
  "warning",
  "error",
  "critical",
  "alert",
  "emergency"
]);
var SetLevelRequestSchema = RequestSchema.extend({
  method: exports_external.literal("logging/setLevel"),
  params: BaseRequestParamsSchema.extend({
    level: LoggingLevelSchema
  })
});
var LoggingMessageNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/message"),
  params: BaseNotificationParamsSchema.extend({
    level: LoggingLevelSchema,
    logger: exports_external.optional(exports_external.string()),
    data: exports_external.unknown()
  })
});
var ModelHintSchema = exports_external.object({
  name: exports_external.string().optional()
}).passthrough();
var ModelPreferencesSchema = exports_external.object({
  hints: exports_external.optional(exports_external.array(ModelHintSchema)),
  costPriority: exports_external.optional(exports_external.number().min(0).max(1)),
  speedPriority: exports_external.optional(exports_external.number().min(0).max(1)),
  intelligencePriority: exports_external.optional(exports_external.number().min(0).max(1))
}).passthrough();
var SamplingMessageSchema = exports_external.object({
  role: exports_external.enum(["user", "assistant"]),
  content: exports_external.union([TextContentSchema, ImageContentSchema])
}).passthrough();
var CreateMessageRequestSchema = RequestSchema.extend({
  method: exports_external.literal("sampling/createMessage"),
  params: BaseRequestParamsSchema.extend({
    messages: exports_external.array(SamplingMessageSchema),
    systemPrompt: exports_external.optional(exports_external.string()),
    includeContext: exports_external.optional(exports_external.enum(["none", "thisServer", "allServers"])),
    temperature: exports_external.optional(exports_external.number()),
    maxTokens: exports_external.number().int(),
    stopSequences: exports_external.optional(exports_external.array(exports_external.string())),
    metadata: exports_external.optional(exports_external.object({}).passthrough()),
    modelPreferences: exports_external.optional(ModelPreferencesSchema)
  })
});
var CreateMessageResultSchema = ResultSchema.extend({
  model: exports_external.string(),
  stopReason: exports_external.optional(exports_external.enum(["endTurn", "stopSequence", "maxTokens"]).or(exports_external.string())),
  role: exports_external.enum(["user", "assistant"]),
  content: exports_external.discriminatedUnion("type", [
    TextContentSchema,
    ImageContentSchema
  ])
});
var ResourceReferenceSchema = exports_external.object({
  type: exports_external.literal("ref/resource"),
  uri: exports_external.string()
}).passthrough();
var PromptReferenceSchema = exports_external.object({
  type: exports_external.literal("ref/prompt"),
  name: exports_external.string()
}).passthrough();
var CompleteRequestSchema = RequestSchema.extend({
  method: exports_external.literal("completion/complete"),
  params: BaseRequestParamsSchema.extend({
    ref: exports_external.union([PromptReferenceSchema, ResourceReferenceSchema]),
    argument: exports_external.object({
      name: exports_external.string(),
      value: exports_external.string()
    }).passthrough()
  })
});
var CompleteResultSchema = ResultSchema.extend({
  completion: exports_external.object({
    values: exports_external.array(exports_external.string()).max(100),
    total: exports_external.optional(exports_external.number().int()),
    hasMore: exports_external.optional(exports_external.boolean())
  }).passthrough()
});
var RootSchema = exports_external.object({
  uri: exports_external.string().startsWith("file://"),
  name: exports_external.optional(exports_external.string())
}).passthrough();
var ListRootsRequestSchema = RequestSchema.extend({
  method: exports_external.literal("roots/list")
});
var ListRootsResultSchema = ResultSchema.extend({
  roots: exports_external.array(RootSchema)
});
var RootsListChangedNotificationSchema = NotificationSchema.extend({
  method: exports_external.literal("notifications/roots/list_changed")
});
var ClientRequestSchema = exports_external.union([
  PingRequestSchema,
  InitializeRequestSchema,
  CompleteRequestSchema,
  SetLevelRequestSchema,
  GetPromptRequestSchema,
  ListPromptsRequestSchema,
  ListResourcesRequestSchema,
  ListResourceTemplatesRequestSchema,
  ReadResourceRequestSchema,
  SubscribeRequestSchema,
  UnsubscribeRequestSchema,
  CallToolRequestSchema,
  ListToolsRequestSchema
]);
var ClientNotificationSchema = exports_external.union([
  CancelledNotificationSchema,
  ProgressNotificationSchema,
  InitializedNotificationSchema,
  RootsListChangedNotificationSchema
]);
var ClientResultSchema = exports_external.union([
  EmptyResultSchema,
  CreateMessageResultSchema,
  ListRootsResultSchema
]);
var ServerRequestSchema = exports_external.union([
  PingRequestSchema,
  CreateMessageRequestSchema,
  ListRootsRequestSchema
]);
var ServerNotificationSchema = exports_external.union([
  CancelledNotificationSchema,
  ProgressNotificationSchema,
  LoggingMessageNotificationSchema,
  ResourceUpdatedNotificationSchema,
  ResourceListChangedNotificationSchema,
  ToolListChangedNotificationSchema,
  PromptListChangedNotificationSchema
]);
var ServerResultSchema = exports_external.union([
  EmptyResultSchema,
  InitializeResultSchema,
  CompleteResultSchema,
  GetPromptResultSchema,
  ListPromptsResultSchema,
  ListResourcesResultSchema,
  ListResourceTemplatesResultSchema,
  ReadResourceResultSchema,
  CallToolResultSchema,
  ListToolsResultSchema
]);

class McpError extends Error {
  constructor(code, message, data) {
    super(`MCP error ${code}: ${message}`);
    this.code = code;
    this.data = data;
  }
}

// node_modules/@modelcontextprotocol/sdk/dist/shared/protocol.js
var DEFAULT_REQUEST_TIMEOUT_MSEC = 60000;

class Protocol {
  constructor(_options) {
    this._options = _options;
    this._requestMessageId = 0;
    this._requestHandlers = new Map;
    this._requestHandlerAbortControllers = new Map;
    this._notificationHandlers = new Map;
    this._responseHandlers = new Map;
    this._progressHandlers = new Map;
    this.setNotificationHandler(CancelledNotificationSchema, (notification) => {
      const controller = this._requestHandlerAbortControllers.get(notification.params.requestId);
      controller === null || controller === undefined || controller.abort(notification.params.reason);
    });
    this.setNotificationHandler(ProgressNotificationSchema, (notification) => {
      this._onprogress(notification);
    });
    this.setRequestHandler(PingRequestSchema, (_request) => ({}));
  }
  async connect(transport) {
    this._transport = transport;
    this._transport.onclose = () => {
      this._onclose();
    };
    this._transport.onerror = (error) => {
      this._onerror(error);
    };
    this._transport.onmessage = (message) => {
      if (!("method" in message)) {
        this._onresponse(message);
      } else if ("id" in message) {
        this._onrequest(message);
      } else {
        this._onnotification(message);
      }
    };
    await this._transport.start();
  }
  _onclose() {
    var _a;
    const responseHandlers = this._responseHandlers;
    this._responseHandlers = new Map;
    this._progressHandlers.clear();
    this._transport = undefined;
    (_a = this.onclose) === null || _a === undefined || _a.call(this);
    const error = new McpError(ErrorCode.ConnectionClosed, "Connection closed");
    for (const handler of responseHandlers.values()) {
      handler(error);
    }
  }
  _onerror(error) {
    var _a;
    (_a = this.onerror) === null || _a === undefined || _a.call(this, error);
  }
  _onnotification(notification) {
    var _a;
    const handler = (_a = this._notificationHandlers.get(notification.method)) !== null && _a !== undefined ? _a : this.fallbackNotificationHandler;
    if (handler === undefined) {
      return;
    }
    Promise.resolve().then(() => handler(notification)).catch((error) => this._onerror(new Error(`Uncaught error in notification handler: ${error}`)));
  }
  _onrequest(request) {
    var _a, _b;
    const handler = (_a = this._requestHandlers.get(request.method)) !== null && _a !== undefined ? _a : this.fallbackRequestHandler;
    if (handler === undefined) {
      (_b = this._transport) === null || _b === undefined || _b.send({
        jsonrpc: "2.0",
        id: request.id,
        error: {
          code: ErrorCode.MethodNotFound,
          message: "Method not found"
        }
      }).catch((error) => this._onerror(new Error(`Failed to send an error response: ${error}`)));
      return;
    }
    const abortController = new AbortController;
    this._requestHandlerAbortControllers.set(request.id, abortController);
    Promise.resolve().then(() => handler(request, { signal: abortController.signal })).then((result) => {
      var _a2;
      if (abortController.signal.aborted) {
        return;
      }
      return (_a2 = this._transport) === null || _a2 === undefined ? undefined : _a2.send({
        result,
        jsonrpc: "2.0",
        id: request.id
      });
    }, (error) => {
      var _a2, _b2;
      if (abortController.signal.aborted) {
        return;
      }
      return (_a2 = this._transport) === null || _a2 === undefined ? undefined : _a2.send({
        jsonrpc: "2.0",
        id: request.id,
        error: {
          code: Number.isSafeInteger(error["code"]) ? error["code"] : ErrorCode.InternalError,
          message: (_b2 = error.message) !== null && _b2 !== undefined ? _b2 : "Internal error"
        }
      });
    }).catch((error) => this._onerror(new Error(`Failed to send response: ${error}`))).finally(() => {
      this._requestHandlerAbortControllers.delete(request.id);
    });
  }
  _onprogress(notification) {
    const { progress, total, progressToken } = notification.params;
    const handler = this._progressHandlers.get(Number(progressToken));
    if (handler === undefined) {
      this._onerror(new Error(`Received a progress notification for an unknown token: ${JSON.stringify(notification)}`));
      return;
    }
    handler({ progress, total });
  }
  _onresponse(response) {
    const messageId = response.id;
    const handler = this._responseHandlers.get(Number(messageId));
    if (handler === undefined) {
      this._onerror(new Error(`Received a response for an unknown message ID: ${JSON.stringify(response)}`));
      return;
    }
    this._responseHandlers.delete(Number(messageId));
    this._progressHandlers.delete(Number(messageId));
    if ("result" in response) {
      handler(response);
    } else {
      const error = new McpError(response.error.code, response.error.message, response.error.data);
      handler(error);
    }
  }
  get transport() {
    return this._transport;
  }
  async close() {
    var _a;
    await ((_a = this._transport) === null || _a === undefined ? undefined : _a.close());
  }
  request(request, resultSchema, options) {
    return new Promise((resolve, reject) => {
      var _a, _b, _c, _d;
      if (!this._transport) {
        reject(new Error("Not connected"));
        return;
      }
      if (((_a = this._options) === null || _a === undefined ? undefined : _a.enforceStrictCapabilities) === true) {
        this.assertCapabilityForMethod(request.method);
      }
      (_b = options === null || options === undefined ? undefined : options.signal) === null || _b === undefined || _b.throwIfAborted();
      const messageId = this._requestMessageId++;
      const jsonrpcRequest = {
        ...request,
        jsonrpc: "2.0",
        id: messageId
      };
      if (options === null || options === undefined ? undefined : options.onprogress) {
        this._progressHandlers.set(messageId, options.onprogress);
        jsonrpcRequest.params = {
          ...request.params,
          _meta: { progressToken: messageId }
        };
      }
      let timeoutId = undefined;
      this._responseHandlers.set(messageId, (response) => {
        var _a2;
        if (timeoutId !== undefined) {
          clearTimeout(timeoutId);
        }
        if ((_a2 = options === null || options === undefined ? undefined : options.signal) === null || _a2 === undefined ? undefined : _a2.aborted) {
          return;
        }
        if (response instanceof Error) {
          return reject(response);
        }
        try {
          const result = resultSchema.parse(response.result);
          resolve(result);
        } catch (error) {
          reject(error);
        }
      });
      const cancel = (reason) => {
        var _a2;
        this._responseHandlers.delete(messageId);
        this._progressHandlers.delete(messageId);
        (_a2 = this._transport) === null || _a2 === undefined || _a2.send({
          jsonrpc: "2.0",
          method: "cancelled",
          params: {
            requestId: messageId,
            reason: String(reason)
          }
        }).catch((error) => this._onerror(new Error(`Failed to send cancellation: ${error}`)));
        reject(reason);
      };
      (_c = options === null || options === undefined ? undefined : options.signal) === null || _c === undefined || _c.addEventListener("abort", () => {
        var _a2;
        if (timeoutId !== undefined) {
          clearTimeout(timeoutId);
        }
        cancel((_a2 = options === null || options === undefined ? undefined : options.signal) === null || _a2 === undefined ? undefined : _a2.reason);
      });
      const timeout = (_d = options === null || options === undefined ? undefined : options.timeout) !== null && _d !== undefined ? _d : DEFAULT_REQUEST_TIMEOUT_MSEC;
      timeoutId = setTimeout(() => cancel(new McpError(ErrorCode.RequestTimeout, "Request timed out", {
        timeout
      })), timeout);
      this._transport.send(jsonrpcRequest).catch((error) => {
        if (timeoutId !== undefined) {
          clearTimeout(timeoutId);
        }
        reject(error);
      });
    });
  }
  async notification(notification) {
    if (!this._transport) {
      throw new Error("Not connected");
    }
    this.assertNotificationCapability(notification.method);
    const jsonrpcNotification = {
      ...notification,
      jsonrpc: "2.0"
    };
    await this._transport.send(jsonrpcNotification);
  }
  setRequestHandler(requestSchema, handler) {
    const method = requestSchema.shape.method.value;
    this.assertRequestHandlerCapability(method);
    this._requestHandlers.set(method, (request, extra) => Promise.resolve(handler(requestSchema.parse(request), extra)));
  }
  removeRequestHandler(method) {
    this._requestHandlers.delete(method);
  }
  setNotificationHandler(notificationSchema, handler) {
    this._notificationHandlers.set(notificationSchema.shape.method.value, (notification) => Promise.resolve(handler(notificationSchema.parse(notification))));
  }
  removeNotificationHandler(method) {
    this._notificationHandlers.delete(method);
  }
}

// node_modules/@modelcontextprotocol/sdk/dist/server/index.js
class Server extends Protocol {
  constructor(_serverInfo, options) {
    super(options);
    this._serverInfo = _serverInfo;
    this._capabilities = options.capabilities;
    this.setRequestHandler(InitializeRequestSchema, (request) => this._oninitialize(request));
    this.setNotificationHandler(InitializedNotificationSchema, () => {
      var _a;
      return (_a = this.oninitialized) === null || _a === undefined ? undefined : _a.call(this);
    });
  }
  assertCapabilityForMethod(method) {
    var _a, _b;
    switch (method) {
      case "sampling/createMessage":
        if (!((_a = this._clientCapabilities) === null || _a === undefined ? undefined : _a.sampling)) {
          throw new Error(`Client does not support sampling (required for ${method})`);
        }
        break;
      case "roots/list":
        if (!((_b = this._clientCapabilities) === null || _b === undefined ? undefined : _b.roots)) {
          throw new Error(`Client does not support listing roots (required for ${method})`);
        }
        break;
      case "ping":
        break;
    }
  }
  assertNotificationCapability(method) {
    switch (method) {
      case "notifications/message":
        if (!this._capabilities.logging) {
          throw new Error(`Server does not support logging (required for ${method})`);
        }
        break;
      case "notifications/resources/updated":
      case "notifications/resources/list_changed":
        if (!this._capabilities.resources) {
          throw new Error(`Server does not support notifying about resources (required for ${method})`);
        }
        break;
      case "notifications/tools/list_changed":
        if (!this._capabilities.tools) {
          throw new Error(`Server does not support notifying of tool list changes (required for ${method})`);
        }
        break;
      case "notifications/prompts/list_changed":
        if (!this._capabilities.prompts) {
          throw new Error(`Server does not support notifying of prompt list changes (required for ${method})`);
        }
        break;
      case "notifications/cancelled":
        break;
      case "notifications/progress":
        break;
    }
  }
  assertRequestHandlerCapability(method) {
    switch (method) {
      case "sampling/createMessage":
        if (!this._capabilities.sampling) {
          throw new Error(`Server does not support sampling (required for ${method})`);
        }
        break;
      case "logging/setLevel":
        if (!this._capabilities.logging) {
          throw new Error(`Server does not support logging (required for ${method})`);
        }
        break;
      case "prompts/get":
      case "prompts/list":
        if (!this._capabilities.prompts) {
          throw new Error(`Server does not support prompts (required for ${method})`);
        }
        break;
      case "resources/list":
      case "resources/templates/list":
      case "resources/read":
        if (!this._capabilities.resources) {
          throw new Error(`Server does not support resources (required for ${method})`);
        }
        break;
      case "tools/call":
      case "tools/list":
        if (!this._capabilities.tools) {
          throw new Error(`Server does not support tools (required for ${method})`);
        }
        break;
      case "ping":
      case "initialize":
        break;
    }
  }
  async _oninitialize(request) {
    const requestedVersion = request.params.protocolVersion;
    this._clientCapabilities = request.params.capabilities;
    this._clientVersion = request.params.clientInfo;
    return {
      protocolVersion: SUPPORTED_PROTOCOL_VERSIONS.includes(requestedVersion) ? requestedVersion : LATEST_PROTOCOL_VERSION,
      capabilities: this.getCapabilities(),
      serverInfo: this._serverInfo
    };
  }
  getClientCapabilities() {
    return this._clientCapabilities;
  }
  getClientVersion() {
    return this._clientVersion;
  }
  getCapabilities() {
    return this._capabilities;
  }
  async ping() {
    return this.request({ method: "ping" }, EmptyResultSchema);
  }
  async createMessage(params, options) {
    return this.request({ method: "sampling/createMessage", params }, CreateMessageResultSchema, options);
  }
  async listRoots(params, options) {
    return this.request({ method: "roots/list", params }, ListRootsResultSchema, options);
  }
  async sendLoggingMessage(params) {
    return this.notification({ method: "notifications/message", params });
  }
  async sendResourceUpdated(params) {
    return this.notification({
      method: "notifications/resources/updated",
      params
    });
  }
  async sendResourceListChanged() {
    return this.notification({
      method: "notifications/resources/list_changed"
    });
  }
  async sendToolListChanged() {
    return this.notification({ method: "notifications/tools/list_changed" });
  }
  async sendPromptListChanged() {
    return this.notification({ method: "notifications/prompts/list_changed" });
  }
}

// node_modules/@modelcontextprotocol/sdk/dist/server/stdio.js
import process2 from "node:process";

// node_modules/@modelcontextprotocol/sdk/dist/shared/stdio.js
class ReadBuffer {
  append(chunk) {
    this._buffer = this._buffer ? Buffer.concat([this._buffer, chunk]) : chunk;
  }
  readMessage() {
    if (!this._buffer) {
      return null;
    }
    const index = this._buffer.indexOf(`
`);
    if (index === -1) {
      return null;
    }
    const line = this._buffer.toString("utf8", 0, index);
    this._buffer = this._buffer.subarray(index + 1);
    return deserializeMessage(line);
  }
  clear() {
    this._buffer = undefined;
  }
}
function deserializeMessage(line) {
  return JSONRPCMessageSchema.parse(JSON.parse(line));
}
function serializeMessage(message) {
  return JSON.stringify(message) + `
`;
}

// node_modules/@modelcontextprotocol/sdk/dist/server/stdio.js
class StdioServerTransport {
  constructor(_stdin = process2.stdin, _stdout = process2.stdout) {
    this._stdin = _stdin;
    this._stdout = _stdout;
    this._readBuffer = new ReadBuffer;
    this._started = false;
    this._ondata = (chunk) => {
      this._readBuffer.append(chunk);
      this.processReadBuffer();
    };
    this._onerror = (error) => {
      var _a;
      (_a = this.onerror) === null || _a === undefined || _a.call(this, error);
    };
  }
  async start() {
    if (this._started) {
      throw new Error("StdioServerTransport already started! If using Server class, note that connect() calls start() automatically.");
    }
    this._started = true;
    this._stdin.on("data", this._ondata);
    this._stdin.on("error", this._onerror);
  }
  processReadBuffer() {
    var _a, _b;
    while (true) {
      try {
        const message = this._readBuffer.readMessage();
        if (message === null) {
          break;
        }
        (_a = this.onmessage) === null || _a === undefined || _a.call(this, message);
      } catch (error) {
        (_b = this.onerror) === null || _b === undefined || _b.call(this, error);
      }
    }
  }
  async close() {
    var _a;
    this._stdin.off("data", this._ondata);
    this._stdin.off("error", this._onerror);
    this._readBuffer.clear();
    (_a = this.onclose) === null || _a === undefined || _a.call(this);
  }
  send(message) {
    return new Promise((resolve) => {
      const json = serializeMessage(message);
      if (this._stdout.write(json)) {
        resolve();
      } else {
        this._stdout.once("drain", resolve);
      }
    });
  }
}

// node_modules/@modelcontextprotocol/sdk/dist/types.js
var JSONRPC_VERSION2 = "2.0";
var ProgressTokenSchema2 = exports_external.union([exports_external.string(), exports_external.number().int()]);
var CursorSchema2 = exports_external.string();
var BaseRequestParamsSchema2 = exports_external.object({
  _meta: exports_external.optional(exports_external.object({
    progressToken: exports_external.optional(ProgressTokenSchema2)
  }).passthrough())
}).passthrough();
var RequestSchema2 = exports_external.object({
  method: exports_external.string(),
  params: exports_external.optional(BaseRequestParamsSchema2)
});
var BaseNotificationParamsSchema2 = exports_external.object({
  _meta: exports_external.optional(exports_external.object({}).passthrough())
}).passthrough();
var NotificationSchema2 = exports_external.object({
  method: exports_external.string(),
  params: exports_external.optional(BaseNotificationParamsSchema2)
});
var ResultSchema2 = exports_external.object({
  _meta: exports_external.optional(exports_external.object({}).passthrough())
}).passthrough();
var RequestIdSchema2 = exports_external.union([exports_external.string(), exports_external.number().int()]);
var JSONRPCRequestSchema2 = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION2),
  id: RequestIdSchema2
}).merge(RequestSchema2).strict();
var JSONRPCNotificationSchema2 = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION2)
}).merge(NotificationSchema2).strict();
var JSONRPCResponseSchema2 = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION2),
  id: RequestIdSchema2,
  result: ResultSchema2
}).strict();
var ErrorCode2;
(function(ErrorCode3) {
  ErrorCode3[ErrorCode3["ConnectionClosed"] = -1] = "ConnectionClosed";
  ErrorCode3[ErrorCode3["RequestTimeout"] = -2] = "RequestTimeout";
  ErrorCode3[ErrorCode3["ParseError"] = -32700] = "ParseError";
  ErrorCode3[ErrorCode3["InvalidRequest"] = -32600] = "InvalidRequest";
  ErrorCode3[ErrorCode3["MethodNotFound"] = -32601] = "MethodNotFound";
  ErrorCode3[ErrorCode3["InvalidParams"] = -32602] = "InvalidParams";
  ErrorCode3[ErrorCode3["InternalError"] = -32603] = "InternalError";
})(ErrorCode2 || (ErrorCode2 = {}));
var JSONRPCErrorSchema2 = exports_external.object({
  jsonrpc: exports_external.literal(JSONRPC_VERSION2),
  id: RequestIdSchema2,
  error: exports_external.object({
    code: exports_external.number().int(),
    message: exports_external.string(),
    data: exports_external.optional(exports_external.unknown())
  })
}).strict();
var JSONRPCMessageSchema2 = exports_external.union([
  JSONRPCRequestSchema2,
  JSONRPCNotificationSchema2,
  JSONRPCResponseSchema2,
  JSONRPCErrorSchema2
]);
var EmptyResultSchema2 = ResultSchema2.strict();
var CancelledNotificationSchema2 = NotificationSchema2.extend({
  method: exports_external.literal("notifications/cancelled"),
  params: BaseNotificationParamsSchema2.extend({
    requestId: RequestIdSchema2,
    reason: exports_external.string().optional()
  })
});
var ImplementationSchema2 = exports_external.object({
  name: exports_external.string(),
  version: exports_external.string()
}).passthrough();
var ClientCapabilitiesSchema2 = exports_external.object({
  experimental: exports_external.optional(exports_external.object({}).passthrough()),
  sampling: exports_external.optional(exports_external.object({}).passthrough()),
  roots: exports_external.optional(exports_external.object({
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough())
}).passthrough();
var InitializeRequestSchema2 = RequestSchema2.extend({
  method: exports_external.literal("initialize"),
  params: BaseRequestParamsSchema2.extend({
    protocolVersion: exports_external.string(),
    capabilities: ClientCapabilitiesSchema2,
    clientInfo: ImplementationSchema2
  })
});
var ServerCapabilitiesSchema2 = exports_external.object({
  experimental: exports_external.optional(exports_external.object({}).passthrough()),
  logging: exports_external.optional(exports_external.object({}).passthrough()),
  prompts: exports_external.optional(exports_external.object({
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough()),
  resources: exports_external.optional(exports_external.object({
    subscribe: exports_external.optional(exports_external.boolean()),
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough()),
  tools: exports_external.optional(exports_external.object({
    listChanged: exports_external.optional(exports_external.boolean())
  }).passthrough())
}).passthrough();
var InitializeResultSchema2 = ResultSchema2.extend({
  protocolVersion: exports_external.string(),
  capabilities: ServerCapabilitiesSchema2,
  serverInfo: ImplementationSchema2
});
var InitializedNotificationSchema2 = NotificationSchema2.extend({
  method: exports_external.literal("notifications/initialized")
});
var PingRequestSchema2 = RequestSchema2.extend({
  method: exports_external.literal("ping")
});
var ProgressSchema2 = exports_external.object({
  progress: exports_external.number(),
  total: exports_external.optional(exports_external.number())
}).passthrough();
var ProgressNotificationSchema2 = NotificationSchema2.extend({
  method: exports_external.literal("notifications/progress"),
  params: BaseNotificationParamsSchema2.merge(ProgressSchema2).extend({
    progressToken: ProgressTokenSchema2
  })
});
var PaginatedRequestSchema2 = RequestSchema2.extend({
  params: BaseRequestParamsSchema2.extend({
    cursor: exports_external.optional(CursorSchema2)
  }).optional()
});
var PaginatedResultSchema2 = ResultSchema2.extend({
  nextCursor: exports_external.optional(CursorSchema2)
});
var ResourceContentsSchema2 = exports_external.object({
  uri: exports_external.string(),
  mimeType: exports_external.optional(exports_external.string())
}).passthrough();
var TextResourceContentsSchema2 = ResourceContentsSchema2.extend({
  text: exports_external.string()
});
var BlobResourceContentsSchema2 = ResourceContentsSchema2.extend({
  blob: exports_external.string().base64()
});
var ResourceSchema2 = exports_external.object({
  uri: exports_external.string(),
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  mimeType: exports_external.optional(exports_external.string())
}).passthrough();
var ResourceTemplateSchema2 = exports_external.object({
  uriTemplate: exports_external.string(),
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  mimeType: exports_external.optional(exports_external.string())
}).passthrough();
var ListResourcesRequestSchema2 = PaginatedRequestSchema2.extend({
  method: exports_external.literal("resources/list")
});
var ListResourcesResultSchema2 = PaginatedResultSchema2.extend({
  resources: exports_external.array(ResourceSchema2)
});
var ListResourceTemplatesRequestSchema2 = PaginatedRequestSchema2.extend({
  method: exports_external.literal("resources/templates/list")
});
var ListResourceTemplatesResultSchema2 = PaginatedResultSchema2.extend({
  resourceTemplates: exports_external.array(ResourceTemplateSchema2)
});
var ReadResourceRequestSchema2 = RequestSchema2.extend({
  method: exports_external.literal("resources/read"),
  params: BaseRequestParamsSchema2.extend({
    uri: exports_external.string()
  })
});
var ReadResourceResultSchema2 = ResultSchema2.extend({
  contents: exports_external.array(exports_external.union([TextResourceContentsSchema2, BlobResourceContentsSchema2]))
});
var ResourceListChangedNotificationSchema2 = NotificationSchema2.extend({
  method: exports_external.literal("notifications/resources/list_changed")
});
var SubscribeRequestSchema2 = RequestSchema2.extend({
  method: exports_external.literal("resources/subscribe"),
  params: BaseRequestParamsSchema2.extend({
    uri: exports_external.string()
  })
});
var UnsubscribeRequestSchema2 = RequestSchema2.extend({
  method: exports_external.literal("resources/unsubscribe"),
  params: BaseRequestParamsSchema2.extend({
    uri: exports_external.string()
  })
});
var ResourceUpdatedNotificationSchema2 = NotificationSchema2.extend({
  method: exports_external.literal("notifications/resources/updated"),
  params: BaseNotificationParamsSchema2.extend({
    uri: exports_external.string()
  })
});
var PromptArgumentSchema2 = exports_external.object({
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  required: exports_external.optional(exports_external.boolean())
}).passthrough();
var PromptSchema2 = exports_external.object({
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  arguments: exports_external.optional(exports_external.array(PromptArgumentSchema2))
}).passthrough();
var ListPromptsRequestSchema2 = PaginatedRequestSchema2.extend({
  method: exports_external.literal("prompts/list")
});
var ListPromptsResultSchema2 = PaginatedResultSchema2.extend({
  prompts: exports_external.array(PromptSchema2)
});
var GetPromptRequestSchema2 = RequestSchema2.extend({
  method: exports_external.literal("prompts/get"),
  params: BaseRequestParamsSchema2.extend({
    name: exports_external.string(),
    arguments: exports_external.optional(exports_external.record(exports_external.string()))
  })
});
var TextContentSchema2 = exports_external.object({
  type: exports_external.literal("text"),
  text: exports_external.string()
}).passthrough();
var ImageContentSchema2 = exports_external.object({
  type: exports_external.literal("image"),
  data: exports_external.string().base64(),
  mimeType: exports_external.string()
}).passthrough();
var EmbeddedResourceSchema2 = exports_external.object({
  type: exports_external.literal("resource"),
  resource: exports_external.union([TextResourceContentsSchema2, BlobResourceContentsSchema2])
}).passthrough();
var PromptMessageSchema2 = exports_external.object({
  role: exports_external.enum(["user", "assistant"]),
  content: exports_external.union([
    TextContentSchema2,
    ImageContentSchema2,
    EmbeddedResourceSchema2
  ])
}).passthrough();
var GetPromptResultSchema2 = ResultSchema2.extend({
  description: exports_external.optional(exports_external.string()),
  messages: exports_external.array(PromptMessageSchema2)
});
var PromptListChangedNotificationSchema2 = NotificationSchema2.extend({
  method: exports_external.literal("notifications/prompts/list_changed")
});
var ToolSchema2 = exports_external.object({
  name: exports_external.string(),
  description: exports_external.optional(exports_external.string()),
  inputSchema: exports_external.object({
    type: exports_external.literal("object"),
    properties: exports_external.optional(exports_external.object({}).passthrough())
  }).passthrough()
}).passthrough();
var ListToolsRequestSchema2 = PaginatedRequestSchema2.extend({
  method: exports_external.literal("tools/list")
});
var ListToolsResultSchema2 = PaginatedResultSchema2.extend({
  tools: exports_external.array(ToolSchema2)
});
var CallToolResultSchema2 = ResultSchema2.extend({
  content: exports_external.array(exports_external.union([TextContentSchema2, ImageContentSchema2, EmbeddedResourceSchema2])),
  isError: exports_external.boolean().default(false).optional()
});
var CompatibilityCallToolResultSchema2 = CallToolResultSchema2.or(ResultSchema2.extend({
  toolResult: exports_external.unknown()
}));
var CallToolRequestSchema2 = RequestSchema2.extend({
  method: exports_external.literal("tools/call"),
  params: BaseRequestParamsSchema2.extend({
    name: exports_external.string(),
    arguments: exports_external.optional(exports_external.record(exports_external.unknown()))
  })
});
var ToolListChangedNotificationSchema2 = NotificationSchema2.extend({
  method: exports_external.literal("notifications/tools/list_changed")
});
var LoggingLevelSchema2 = exports_external.enum([
  "debug",
  "info",
  "notice",
  "warning",
  "error",
  "critical",
  "alert",
  "emergency"
]);
var SetLevelRequestSchema2 = RequestSchema2.extend({
  method: exports_external.literal("logging/setLevel"),
  params: BaseRequestParamsSchema2.extend({
    level: LoggingLevelSchema2
  })
});
var LoggingMessageNotificationSchema2 = NotificationSchema2.extend({
  method: exports_external.literal("notifications/message"),
  params: BaseNotificationParamsSchema2.extend({
    level: LoggingLevelSchema2,
    logger: exports_external.optional(exports_external.string()),
    data: exports_external.unknown()
  })
});
var ModelHintSchema2 = exports_external.object({
  name: exports_external.string().optional()
}).passthrough();
var ModelPreferencesSchema2 = exports_external.object({
  hints: exports_external.optional(exports_external.array(ModelHintSchema2)),
  costPriority: exports_external.optional(exports_external.number().min(0).max(1)),
  speedPriority: exports_external.optional(exports_external.number().min(0).max(1)),
  intelligencePriority: exports_external.optional(exports_external.number().min(0).max(1))
}).passthrough();
var SamplingMessageSchema2 = exports_external.object({
  role: exports_external.enum(["user", "assistant"]),
  content: exports_external.union([TextContentSchema2, ImageContentSchema2])
}).passthrough();
var CreateMessageRequestSchema2 = RequestSchema2.extend({
  method: exports_external.literal("sampling/createMessage"),
  params: BaseRequestParamsSchema2.extend({
    messages: exports_external.array(SamplingMessageSchema2),
    systemPrompt: exports_external.optional(exports_external.string()),
    includeContext: exports_external.optional(exports_external.enum(["none", "thisServer", "allServers"])),
    temperature: exports_external.optional(exports_external.number()),
    maxTokens: exports_external.number().int(),
    stopSequences: exports_external.optional(exports_external.array(exports_external.string())),
    metadata: exports_external.optional(exports_external.object({}).passthrough()),
    modelPreferences: exports_external.optional(ModelPreferencesSchema2)
  })
});
var CreateMessageResultSchema2 = ResultSchema2.extend({
  model: exports_external.string(),
  stopReason: exports_external.optional(exports_external.enum(["endTurn", "stopSequence", "maxTokens"]).or(exports_external.string())),
  role: exports_external.enum(["user", "assistant"]),
  content: exports_external.discriminatedUnion("type", [
    TextContentSchema2,
    ImageContentSchema2
  ])
});
var ResourceReferenceSchema2 = exports_external.object({
  type: exports_external.literal("ref/resource"),
  uri: exports_external.string()
}).passthrough();
var PromptReferenceSchema2 = exports_external.object({
  type: exports_external.literal("ref/prompt"),
  name: exports_external.string()
}).passthrough();
var CompleteRequestSchema2 = RequestSchema2.extend({
  method: exports_external.literal("completion/complete"),
  params: BaseRequestParamsSchema2.extend({
    ref: exports_external.union([PromptReferenceSchema2, ResourceReferenceSchema2]),
    argument: exports_external.object({
      name: exports_external.string(),
      value: exports_external.string()
    }).passthrough()
  })
});
var CompleteResultSchema2 = ResultSchema2.extend({
  completion: exports_external.object({
    values: exports_external.array(exports_external.string()).max(100),
    total: exports_external.optional(exports_external.number().int()),
    hasMore: exports_external.optional(exports_external.boolean())
  }).passthrough()
});
var RootSchema2 = exports_external.object({
  uri: exports_external.string().startsWith("file://"),
  name: exports_external.optional(exports_external.string())
}).passthrough();
var ListRootsRequestSchema2 = RequestSchema2.extend({
  method: exports_external.literal("roots/list")
});
var ListRootsResultSchema2 = ResultSchema2.extend({
  roots: exports_external.array(RootSchema2)
});
var RootsListChangedNotificationSchema2 = NotificationSchema2.extend({
  method: exports_external.literal("notifications/roots/list_changed")
});
var ClientRequestSchema2 = exports_external.union([
  PingRequestSchema2,
  InitializeRequestSchema2,
  CompleteRequestSchema2,
  SetLevelRequestSchema2,
  GetPromptRequestSchema2,
  ListPromptsRequestSchema2,
  ListResourcesRequestSchema2,
  ListResourceTemplatesRequestSchema2,
  ReadResourceRequestSchema2,
  SubscribeRequestSchema2,
  UnsubscribeRequestSchema2,
  CallToolRequestSchema2,
  ListToolsRequestSchema2
]);
var ClientNotificationSchema2 = exports_external.union([
  CancelledNotificationSchema2,
  ProgressNotificationSchema2,
  InitializedNotificationSchema2,
  RootsListChangedNotificationSchema2
]);
var ClientResultSchema2 = exports_external.union([
  EmptyResultSchema2,
  CreateMessageResultSchema2,
  ListRootsResultSchema2
]);
var ServerRequestSchema2 = exports_external.union([
  PingRequestSchema2,
  CreateMessageRequestSchema2,
  ListRootsRequestSchema2
]);
var ServerNotificationSchema2 = exports_external.union([
  CancelledNotificationSchema2,
  ProgressNotificationSchema2,
  LoggingMessageNotificationSchema2,
  ResourceUpdatedNotificationSchema2,
  ResourceListChangedNotificationSchema2,
  ToolListChangedNotificationSchema2,
  PromptListChangedNotificationSchema2
]);
var ServerResultSchema2 = exports_external.union([
  EmptyResultSchema2,
  InitializeResultSchema2,
  CompleteResultSchema2,
  GetPromptResultSchema2,
  ListPromptsResultSchema2,
  ListResourcesResultSchema2,
  ListResourceTemplatesResultSchema2,
  ReadResourceResultSchema2,
  CallToolResultSchema2,
  ListToolsResultSchema2
]);

// src/index.ts
import { existsSync } from "fs";
import { dirname, resolve } from "path";
import { fileURLToPath } from "url";

// src/tools/index.ts
var movingAverageTools = [
  {
    name: "indicator_sma",
    description: "Simple Moving Average - arithmetic mean over period",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_ema",
    description: "Exponential Moving Average - weighted toward recent prices",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_wma",
    description: "Weighted Moving Average - linearly weighted",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_hma",
    description: "Hull Moving Average - fast, reduced-lag",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_dema",
    description: "Double Exponential Moving Average",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_tema",
    description: "Triple Exponential Moving Average",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_rma",
    description: "Running Moving Average (Wilder's smoothing)",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_vidya",
    description: "Variable Index Dynamic Average",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_ama",
    description: "Adaptive Moving Average (Kaufman's)",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" },
        fast_period: { type: "number" },
        slow_period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_vwap",
    description: "Volume Weighted Average Price",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } }
      },
      required: ["bars"]
    }
  }
];
var momentumTools = [
  {
    name: "indicator_rsi",
    description: "Relative Strength Index (0-100)",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_macd",
    description: "Moving Average Convergence Divergence",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        fast: { type: "number", default: 12 },
        slow: { type: "number", default: 26 },
        signal: { type: "number", default: 9 }
      },
      required: ["prices"]
    }
  },
  {
    name: "indicator_cci",
    description: "Commodity Channel Index",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        period: { type: "number" }
      },
      required: ["bars", "period"]
    }
  },
  {
    name: "indicator_stochastic",
    description: "Stochastic Oscillator %K/%D",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        k_period: { type: "number" },
        d_period: { type: "number" }
      },
      required: ["bars", "k_period", "d_period"]
    }
  },
  {
    name: "indicator_roc",
    description: "Rate of Change",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_obv",
    description: "On-Balance Volume",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } }
      },
      required: ["bars"]
    }
  },
  {
    name: "indicator_aroon",
    description: "Aroon Up/Down oscillator",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        period: { type: "number" }
      },
      required: ["bars", "period"]
    }
  },
  {
    name: "indicator_cmo",
    description: "Chande Momentum Oscillator",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_williams_r",
    description: "Williams %R",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        period: { type: "number" }
      },
      required: ["bars", "period"]
    }
  },
  {
    name: "indicator_mfi",
    description: "Money Flow Index",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        period: { type: "number" }
      },
      required: ["bars", "period"]
    }
  },
  {
    name: "indicator_kvo",
    description: "Klinger Volume Oscillator",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } }
      },
      required: ["bars"]
    }
  },
  {
    name: "indicator_adx",
    description: "Average Directional Index",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        period: { type: "number" }
      },
      required: ["bars", "period"]
    }
  },
  {
    name: "indicator_di",
    description: "Directional Indicator (+DI/-DI)",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        period: { type: "number" }
      },
      required: ["bars", "period"]
    }
  },
  {
    name: "indicator_ppo",
    description: "Percentage Price Oscillator",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        fast: { type: "number" },
        slow: { type: "number" }
      },
      required: ["prices", "fast", "slow"]
    }
  },
  {
    name: "indicator_trix",
    description: "Triple Smoothed EMA Rate of Change",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_psl",
    description: "Psychological Line",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_vhf",
    description: "Vertical Horizontal Filter (trend strength)",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_bias",
    description: "Price deviation from moving average",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  }
];
var volatilityTools = [
  {
    name: "indicator_atr",
    description: "Average True Range",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        period: { type: "number" }
      },
      required: ["bars", "period"]
    }
  },
  {
    name: "indicator_bollinger",
    description: "Bollinger Bands",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" },
        std_dev: { type: "number", default: 2 }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_keltner",
    description: "Keltner Channel",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        ema_period: { type: "number" },
        atr_period: { type: "number" },
        multiplier: { type: "number" }
      },
      required: ["bars", "ema_period", "atr_period", "multiplier"]
    }
  },
  {
    name: "indicator_donchian",
    description: "Donchian Channel",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        period: { type: "number" }
      },
      required: ["bars", "period"]
    }
  },
  {
    name: "indicator_natr",
    description: "Normalized Average True Range",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        period: { type: "number" }
      },
      required: ["bars", "period"]
    }
  },
  {
    name: "indicator_rvi",
    description: "Relative Volatility Index",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_vr",
    description: "Volatility Ratio",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        period: { type: "number" }
      },
      required: ["bars", "period"]
    }
  },
  {
    name: "indicator_bbwidth",
    description: "Bollinger Band Width",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" },
        std_dev: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "indicator_chaikin_vol",
    description: "Chaikin Volatility",
    inputSchema: {
      type: "object",
      properties: {
        bars: { type: "array", items: { type: "object" } },
        period: { type: "number" }
      },
      required: ["bars", "period"]
    }
  },
  {
    name: "indicator_fuzzy_regime",
    description: "Fuzzy volatility regime classification",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  }
];
var riskTools = [
  {
    name: "risk_var_parametric",
    description: "Value at Risk (Gaussian)",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } },
        confidence: { type: "number" }
      },
      required: ["returns", "confidence"]
    }
  },
  {
    name: "risk_var_historical",
    description: "Value at Risk (Historical)",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } },
        confidence: { type: "number" }
      },
      required: ["returns", "confidence"]
    }
  },
  {
    name: "risk_var_monte_carlo",
    description: "Value at Risk (Monte Carlo simulation)",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } },
        confidence: { type: "number" },
        simulations: { type: "number", default: 1e4 }
      },
      required: ["returns", "confidence"]
    }
  },
  {
    name: "risk_cvar",
    description: "Conditional VaR / Expected Shortfall",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } },
        confidence: { type: "number" }
      },
      required: ["returns", "confidence"]
    }
  },
  {
    name: "risk_kelly_criterion",
    description: "Kelly Criterion: f* = (p*b - q) / b",
    inputSchema: {
      type: "object",
      properties: {
        win_rate: { type: "number" },
        win_loss_ratio: { type: "number" }
      },
      required: ["win_rate", "win_loss_ratio"]
    }
  },
  {
    name: "risk_position_size",
    description: "Fixed-risk position sizing",
    inputSchema: {
      type: "object",
      properties: {
        equity: { type: "number" },
        risk_per_trade: { type: "number" },
        entry_price: { type: "number" },
        stop_loss: { type: "number" }
      },
      required: ["equity", "risk_per_trade", "entry_price", "stop_loss"]
    }
  },
  {
    name: "risk_max_drawdown",
    description: "Maximum Drawdown",
    inputSchema: {
      type: "object",
      properties: {
        equity_curve: { type: "array", items: { type: "number" } }
      },
      required: ["equity_curve"]
    }
  },
  {
    name: "risk_drawdown_duration",
    description: "Drawdown duration analysis",
    inputSchema: {
      type: "object",
      properties: {
        equity_curve: { type: "array", items: { type: "number" } }
      },
      required: ["equity_curve"]
    }
  },
  {
    name: "risk_hurst_exponent",
    description: "Hurst Exponent (trend/mean-reversion)",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } }
      },
      required: ["prices"]
    }
  },
  {
    name: "risk_tail_ratio",
    description: "Tail Risk Ratio",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } }
      },
      required: ["returns"]
    }
  },
  {
    name: "risk_omega_ratio",
    description: "Omega Ratio",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } },
        threshold: { type: "number", default: 0 }
      },
      required: ["returns"]
    }
  },
  {
    name: "risk_ulcer_index",
    description: "Ulcer Index (downside volatility)",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "risk_gain_to_pain",
    description: "Gain-to-Pain Ratio",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } }
      },
      required: ["returns"]
    }
  },
  {
    name: "risk_kurtosis",
    description: "Excess Kurtosis (fat tails)",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } }
      },
      required: ["returns"]
    }
  },
  {
    name: "risk_skewness",
    description: "Return Distribution Skewness",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } }
      },
      required: ["returns"]
    }
  },
  {
    name: "risk_volatility_cone",
    description: "Volatility Cone Analysis",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        windows: { type: "array", items: { type: "number" } }
      },
      required: ["prices", "windows"]
    }
  },
  {
    name: "risk_correlation_breakdown",
    description: "Correlation breakdown detection",
    inputSchema: {
      type: "object",
      properties: {
        asset1_returns: { type: "array", items: { type: "number" } },
        asset2_returns: { type: "array", items: { type: "number" } },
        window: { type: "number" }
      },
      required: ["asset1_returns", "asset2_returns", "window"]
    }
  },
  {
    name: "risk_beta",
    description: "Market Beta",
    inputSchema: {
      type: "object",
      properties: {
        asset_returns: { type: "array", items: { type: "number" } },
        market_returns: { type: "array", items: { type: "number" } }
      },
      required: ["asset_returns", "market_returns"]
    }
  },
  {
    name: "risk_tracking_error",
    description: "Tracking Error vs benchmark",
    inputSchema: {
      type: "object",
      properties: {
        portfolio_returns: { type: "array", items: { type: "number" } },
        benchmark_returns: { type: "array", items: { type: "number" } }
      },
      required: ["portfolio_returns", "benchmark_returns"]
    }
  },
  {
    name: "risk_active_share",
    description: "Active Share vs index",
    inputSchema: {
      type: "object",
      properties: {
        portfolio_weights: { type: "array", items: { type: "number" } },
        benchmark_weights: { type: "array", items: { type: "number" } }
      },
      required: ["portfolio_weights", "benchmark_weights"]
    }
  }
];
var portfolioTools = [
  {
    name: "portfolio_sharpe",
    description: "Sharpe Ratio: (R - Rf) / σ",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } },
        risk_free_rate: { type: "number", default: 0 }
      },
      required: ["returns"]
    }
  },
  {
    name: "portfolio_sortino",
    description: "Sortino Ratio (downside deviation)",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } },
        risk_free_rate: { type: "number" },
        target: { type: "number", default: 0 }
      },
      required: ["returns"]
    }
  },
  {
    name: "portfolio_calmar",
    description: "Calmar Ratio: CAGR / MaxDD",
    inputSchema: {
      type: "object",
      properties: {
        equity_curve: { type: "array", items: { type: "number" } },
        periods_per_year: { type: "number" }
      },
      required: ["equity_curve", "periods_per_year"]
    }
  },
  {
    name: "portfolio_cagr",
    description: "Compound Annual Growth Rate",
    inputSchema: {
      type: "object",
      properties: {
        equity_curve: { type: "array", items: { type: "number" } },
        periods_per_year: { type: "number" }
      },
      required: ["equity_curve", "periods_per_year"]
    }
  },
  {
    name: "portfolio_win_rate",
    description: "Win Rate percentage",
    inputSchema: {
      type: "object",
      properties: {
        trade_pnls: { type: "array", items: { type: "number" } }
      },
      required: ["trade_pnls"]
    }
  },
  {
    name: "portfolio_profit_factor",
    description: "Profit Factor: gross profit / gross loss",
    inputSchema: {
      type: "object",
      properties: {
        trade_pnls: { type: "array", items: { type: "number" } }
      },
      required: ["trade_pnls"]
    }
  },
  {
    name: "portfolio_expectancy",
    description: "Trade Expectancy",
    inputSchema: {
      type: "object",
      properties: {
        trade_pnls: { type: "array", items: { type: "number" } }
      },
      required: ["trade_pnls"]
    }
  },
  {
    name: "portfolio_risk_return",
    description: "Risk/Return Ratio",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } }
      },
      required: ["returns"]
    }
  },
  {
    name: "portfolio_avg_win",
    description: "Average Winning Trade",
    inputSchema: {
      type: "object",
      properties: {
        trade_pnls: { type: "array", items: { type: "number" } }
      },
      required: ["trade_pnls"]
    }
  },
  {
    name: "portfolio_avg_loss",
    description: "Average Losing Trade",
    inputSchema: {
      type: "object",
      properties: {
        trade_pnls: { type: "array", items: { type: "number" } }
      },
      required: ["trade_pnls"]
    }
  },
  {
    name: "portfolio_long_ratio",
    description: "Long Exposure Ratio",
    inputSchema: {
      type: "object",
      properties: {
        positions: { type: "array", items: { type: "object" } }
      },
      required: ["positions"]
    }
  },
  {
    name: "portfolio_returns_volatility",
    description: "Returns Standard Deviation",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } }
      },
      required: ["returns"]
    }
  },
  {
    name: "portfolio_information_ratio",
    description: "Information Ratio vs benchmark",
    inputSchema: {
      type: "object",
      properties: {
        portfolio_returns: { type: "array", items: { type: "number" } },
        benchmark_returns: { type: "array", items: { type: "number" } }
      },
      required: ["portfolio_returns", "benchmark_returns"]
    }
  },
  {
    name: "portfolio_treynor",
    description: "Treynor Ratio: (R - Rf) / β",
    inputSchema: {
      type: "object",
      properties: {
        portfolio_returns: { type: "array", items: { type: "number" } },
        market_returns: { type: "array", items: { type: "number" } },
        risk_free_rate: { type: "number" }
      },
      required: ["portfolio_returns", "market_returns"]
    }
  },
  {
    name: "portfolio_alpha",
    description: "Jensen's Alpha",
    inputSchema: {
      type: "object",
      properties: {
        portfolio_returns: { type: "array", items: { type: "number" } },
        market_returns: { type: "array", items: { type: "number" } },
        risk_free_rate: { type: "number" }
      },
      required: ["portfolio_returns", "market_returns"]
    }
  },
  {
    name: "portfolio_correlation_matrix",
    description: "Asset Correlation Matrix",
    inputSchema: {
      type: "object",
      properties: {
        returns_matrix: { type: "array", items: { type: "array" } }
      },
      required: ["returns_matrix"]
    }
  },
  {
    name: "portfolio_covariance_matrix",
    description: "Asset Covariance Matrix",
    inputSchema: {
      type: "object",
      properties: {
        returns_matrix: { type: "array", items: { type: "array" } }
      },
      required: ["returns_matrix"]
    }
  },
  {
    name: "portfolio_efficient_frontier",
    description: "Markowitz Efficient Frontier",
    inputSchema: {
      type: "object",
      properties: {
        expected_returns: { type: "array", items: { type: "number" } },
        covariance_matrix: { type: "array", items: { type: "array" } },
        num_portfolios: { type: "number", default: 100 }
      },
      required: ["expected_returns", "covariance_matrix"]
    }
  },
  {
    name: "portfolio_max_diversification",
    description: "Maximum Diversification Portfolio",
    inputSchema: {
      type: "object",
      properties: {
        expected_returns: { type: "array", items: { type: "number" } },
        covariance_matrix: { type: "array", items: { type: "array" } }
      },
      required: ["expected_returns", "covariance_matrix"]
    }
  },
  {
    name: "portfolio_risk_parity",
    description: "Risk Parity Portfolio",
    inputSchema: {
      type: "object",
      properties: {
        covariance_matrix: { type: "array", items: { type: "array" } }
      },
      required: ["covariance_matrix"]
    }
  },
  {
    name: "portfolio_min_variance",
    description: "Minimum Variance Portfolio",
    inputSchema: {
      type: "object",
      properties: {
        covariance_matrix: { type: "array", items: { type: "array" } }
      },
      required: ["covariance_matrix"]
    }
  }
];
var executionTools = [
  {
    name: "execution_vwap_slippage",
    description: "VWAP Slippage Analysis",
    inputSchema: {
      type: "object",
      properties: {
        trades: { type: "array", items: { type: "object" } },
        vwap_benchmark: { type: "number" }
      },
      required: ["trades", "vwap_benchmark"]
    }
  },
  {
    name: "execution_twap",
    description: "Time-Weighted Average Price",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } }
      },
      required: ["prices"]
    }
  },
  {
    name: "execution_arrival_price",
    description: "Arrival Price Benchmark",
    inputSchema: {
      type: "object",
      properties: {
        trades: { type: "array", items: { type: "object" } },
        arrival_price: { type: "number" }
      },
      required: ["trades", "arrival_price"]
    }
  },
  {
    name: "execution_implementation_shortfall",
    description: "Implementation Shortfall",
    inputSchema: {
      type: "object",
      properties: {
        trades: { type: "array", items: { type: "object" } },
        decision_price: { type: "number" }
      },
      required: ["trades", "decision_price"]
    }
  },
  {
    name: "execution_market_impact",
    description: "Estimated Market Impact",
    inputSchema: {
      type: "object",
      properties: {
        order_size: { type: "number" },
        avg_daily_volume: { type: "number" },
        volatility: { type: "number" }
      },
      required: ["order_size", "avg_daily_volume", "volatility"]
    }
  },
  {
    name: "execution_spread_cost",
    description: "Bid-Ask Spread Cost",
    inputSchema: {
      type: "object",
      properties: {
        bid: { type: "number" },
        ask: { type: "number" },
        quantity: { type: "number" }
      },
      required: ["bid", "ask", "quantity"]
    }
  },
  {
    name: "orderflow_imbalance",
    description: "Order Flow Imbalance",
    inputSchema: {
      type: "object",
      properties: {
        trades: { type: "array", items: { type: "object" } }
      },
      required: ["trades"]
    }
  },
  {
    name: "orderflow_toxicity",
    description: "VPIN (Volume-synchronized PIN)",
    inputSchema: {
      type: "object",
      properties: {
        trades: { type: "array", items: { type: "object" } },
        bucket_size: { type: "number" }
      },
      required: ["trades", "bucket_size"]
    }
  },
  {
    name: "orderflow_kyle_lambda",
    description: "Kyle's Lambda (price impact)",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        volumes: { type: "array", items: { type: "number" } }
      },
      required: ["prices", "volumes"]
    }
  },
  {
    name: "orderflow_amihud",
    description: "Amihud Illiquidity Ratio",
    inputSchema: {
      type: "object",
      properties: {
        returns: { type: "array", items: { type: "number" } },
        volumes: { type: "array", items: { type: "number" } }
      },
      required: ["returns", "volumes"]
    }
  },
  {
    name: "orderflow_tick_test",
    description: "Tick Test for trade classification",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } }
      },
      required: ["prices"]
    }
  },
  {
    name: "orderflow_quote_rule",
    description: "Quote Rule for trade classification",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        bids: { type: "array", items: { type: "number" } },
        asks: { type: "array", items: { type: "number" } }
      },
      required: ["prices", "bids", "asks"]
    }
  }
];
var regimeTools = [
  {
    name: "regime_pbit_state",
    description: "pBit Market State (Boltzmann)",
    inputSchema: {
      type: "object",
      properties: {
        market_signal: { type: "number" },
        volatility: { type: "number" },
        temperature: { type: "number" }
      },
      required: ["market_signal", "volatility", "temperature"]
    }
  },
  {
    name: "regime_ising_energy",
    description: "Ising Model Market Coherence",
    inputSchema: {
      type: "object",
      properties: {
        asset_returns: { type: "array", items: { type: "number" } }
      },
      required: ["asset_returns"]
    }
  },
  {
    name: "regime_hyperbolic_embed",
    description: "Hyperbolic Market Embedding (Lorentz)",
    inputSchema: {
      type: "object",
      properties: {
        features: { type: "array", items: { type: "number" } }
      },
      required: ["features"]
    }
  },
  {
    name: "regime_lorentz_distance",
    description: "Hyperbolic distance between states",
    inputSchema: {
      type: "object",
      properties: {
        state1: { type: "array", items: { type: "number" } },
        state2: { type: "array", items: { type: "number" } }
      },
      required: ["state1", "state2"]
    }
  },
  {
    name: "regime_boltzmann_dist",
    description: "Boltzmann Distribution of States",
    inputSchema: {
      type: "object",
      properties: {
        energies: { type: "array", items: { type: "number" } },
        temperature: { type: "number" }
      },
      required: ["energies", "temperature"]
    }
  },
  {
    name: "regime_critical_temp",
    description: "Proximity to Critical Temperature",
    inputSchema: {
      type: "object",
      properties: {
        market_data: { type: "array", items: { type: "number" } }
      },
      required: ["market_data"]
    }
  },
  {
    name: "regime_hmm_filter",
    description: "Hidden Markov Model Filtering",
    inputSchema: {
      type: "object",
      properties: {
        observations: { type: "array", items: { type: "number" } },
        n_states: { type: "number" }
      },
      required: ["observations", "n_states"]
    }
  },
  {
    name: "regime_change_detection",
    description: "Online Regime Change Detection",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        window: { type: "number" }
      },
      required: ["prices", "window"]
    }
  },
  {
    name: "regime_volatility_state",
    description: "Bull/Bear/Sideways Classification",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        period: { type: "number" }
      },
      required: ["prices", "period"]
    }
  },
  {
    name: "regime_phi_coherence",
    description: "Integrated Information (Φ) in Markets",
    inputSchema: {
      type: "object",
      properties: {
        asset_returns: { type: "array", items: { type: "array" } }
      },
      required: ["asset_returns"]
    }
  },
  {
    name: "regime_entropy",
    description: "Market Entropy (disorder measure)",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } },
        window: { type: "number" }
      },
      required: ["prices", "window"]
    }
  },
  {
    name: "regime_fractal_dimension",
    description: "Fractal Dimension of Price Series",
    inputSchema: {
      type: "object",
      properties: {
        prices: { type: "array", items: { type: "number" } }
      },
      required: ["prices"]
    }
  }
];
var conformalTools = [
  {
    name: "conformal_prediction_interval",
    description: "Conformal Prediction Interval",
    inputSchema: {
      type: "object",
      properties: {
        residuals: { type: "array", items: { type: "number" } },
        prediction: { type: "number" },
        confidence: { type: "number" }
      },
      required: ["residuals", "prediction", "confidence"]
    }
  },
  {
    name: "conformal_quantile_regression",
    description: "Conformalized Quantile Regression",
    inputSchema: {
      type: "object",
      properties: {
        residuals: { type: "array", items: { type: "number" } },
        quantiles: { type: "array", items: { type: "number" } },
        prediction: { type: "number" }
      },
      required: ["residuals", "quantiles", "prediction"]
    }
  },
  {
    name: "conformal_calibration",
    description: "Calibration Score Computation",
    inputSchema: {
      type: "object",
      properties: {
        predictions: { type: "array", items: { type: "number" } },
        actuals: { type: "array", items: { type: "number" } }
      },
      required: ["predictions", "actuals"]
    }
  },
  {
    name: "conformal_coverage_test",
    description: "Coverage Validation Test",
    inputSchema: {
      type: "object",
      properties: {
        intervals: { type: "array", items: { type: "object" } },
        actuals: { type: "array", items: { type: "number" } },
        target_coverage: { type: "number" }
      },
      required: ["intervals", "actuals", "target_coverage"]
    }
  },
  {
    name: "conformal_width_analysis",
    description: "Prediction Interval Width Analysis",
    inputSchema: {
      type: "object",
      properties: {
        intervals: { type: "array", items: { type: "object" } }
      },
      required: ["intervals"]
    }
  },
  {
    name: "conformal_adaptive_interval",
    description: "Adaptive Conformal Intervals",
    inputSchema: {
      type: "object",
      properties: {
        residuals: { type: "array", items: { type: "number" } },
        prediction: { type: "number" },
        confidence: { type: "number" },
        decay: { type: "number" }
      },
      required: ["residuals", "prediction", "confidence"]
    }
  },
  {
    name: "conformal_multi_horizon",
    description: "Multi-Horizon Prediction Intervals",
    inputSchema: {
      type: "object",
      properties: {
        residuals_by_horizon: { type: "array", items: { type: "array" } },
        predictions: { type: "array", items: { type: "number" } },
        confidence: { type: "number" }
      },
      required: ["residuals_by_horizon", "predictions", "confidence"]
    }
  },
  {
    name: "conformal_probabilistic",
    description: "Probabilistic Predictions",
    inputSchema: {
      type: "object",
      properties: {
        residuals: { type: "array", items: { type: "number" } },
        prediction: { type: "number" },
        n_quantiles: { type: "number", default: 10 }
      },
      required: ["residuals", "prediction"]
    }
  },
  {
    name: "conformal_classification",
    description: "Conformal Classification Sets",
    inputSchema: {
      type: "object",
      properties: {
        scores: { type: "array", items: { type: "array" } },
        confidence: { type: "number" }
      },
      required: ["scores", "confidence"]
    }
  },
  {
    name: "conformal_split_validate",
    description: "Split Conformal Validation",
    inputSchema: {
      type: "object",
      properties: {
        predictions: { type: "array", items: { type: "number" } },
        actuals: { type: "array", items: { type: "number" } },
        split_ratio: { type: "number", default: 0.5 }
      },
      required: ["predictions", "actuals"]
    }
  }
];
var greeksTools = [
  {
    name: "greeks_delta",
    description: "Delta: ∂V/∂S",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" },
        is_call: { type: "boolean" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"]
    }
  },
  {
    name: "greeks_gamma",
    description: "Gamma: ∂²V/∂S²",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry"]
    }
  },
  {
    name: "greeks_theta",
    description: "Theta: ∂V/∂t (daily)",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" },
        is_call: { type: "boolean" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"]
    }
  },
  {
    name: "greeks_vega",
    description: "Vega: ∂V/∂σ (per 1%)",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry"]
    }
  },
  {
    name: "greeks_rho",
    description: "Rho: ∂V/∂r",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" },
        is_call: { type: "boolean" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"]
    }
  },
  {
    name: "greeks_vanna",
    description: "Vanna: ∂²V/∂S∂σ",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry"]
    }
  },
  {
    name: "greeks_volga",
    description: "Volga: ∂²V/∂σ²",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry"]
    }
  },
  {
    name: "greeks_charm",
    description: "Charm: ∂²V/∂S∂t",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" },
        is_call: { type: "boolean" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"]
    }
  },
  {
    name: "options_black_scholes",
    description: "Black-Scholes Pricing",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" },
        is_call: { type: "boolean" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"]
    }
  },
  {
    name: "options_implied_vol",
    description: "Implied Volatility Solver",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        time_to_expiry: { type: "number" },
        is_call: { type: "boolean" },
        market_price: { type: "number" }
      },
      required: ["spot", "strike", "rate", "time_to_expiry", "is_call", "market_price"]
    }
  },
  {
    name: "options_put_call_parity",
    description: "Put-Call Parity Check",
    inputSchema: {
      type: "object",
      properties: {
        call_price: { type: "number" },
        put_price: { type: "number" },
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        time_to_expiry: { type: "number" }
      },
      required: ["call_price", "put_price", "spot", "strike", "rate", "time_to_expiry"]
    }
  },
  {
    name: "options_binomial",
    description: "Binomial Tree Pricing",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" },
        steps: { type: "number" },
        is_call: { type: "boolean" },
        is_american: { type: "boolean" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "steps", "is_call"]
    }
  },
  {
    name: "options_monte_carlo",
    description: "Monte Carlo Option Pricing",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" },
        simulations: { type: "number", default: 1e4 },
        is_call: { type: "boolean" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"]
    }
  },
  {
    name: "options_american",
    description: "American Option Pricing (LSM)",
    inputSchema: {
      type: "object",
      properties: {
        spot: { type: "number" },
        strike: { type: "number" },
        rate: { type: "number" },
        volatility: { type: "number" },
        time_to_expiry: { type: "number" },
        is_call: { type: "boolean" }
      },
      required: ["spot", "strike", "rate", "volatility", "time_to_expiry", "is_call"]
    }
  },
  {
    name: "options_surface_fit",
    description: "Volatility Surface Fitting",
    inputSchema: {
      type: "object",
      properties: {
        strikes: { type: "array", items: { type: "number" } },
        expiries: { type: "array", items: { type: "number" } },
        implied_vols: { type: "array", items: { type: "array" } }
      },
      required: ["strikes", "expiries", "implied_vols"]
    }
  },
  {
    name: "options_skew",
    description: "Volatility Skew Analysis",
    inputSchema: {
      type: "object",
      properties: {
        strikes: { type: "array", items: { type: "number" } },
        implied_vols: { type: "array", items: { type: "number" } },
        spot: { type: "number" }
      },
      required: ["strikes", "implied_vols", "spot"]
    }
  },
  {
    name: "options_term_structure",
    description: "Volatility Term Structure",
    inputSchema: {
      type: "object",
      properties: {
        expiries: { type: "array", items: { type: "number" } },
        implied_vols: { type: "array", items: { type: "number" } }
      },
      required: ["expiries", "implied_vols"]
    }
  },
  {
    name: "options_greeks_portfolio",
    description: "Portfolio Greeks Aggregation",
    inputSchema: {
      type: "object",
      properties: {
        positions: { type: "array", items: { type: "object" } },
        spot: { type: "number" },
        rate: { type: "number" }
      },
      required: ["positions", "spot", "rate"]
    }
  },
  {
    name: "options_hedge_ratio",
    description: "Delta Hedge Ratio",
    inputSchema: {
      type: "object",
      properties: {
        option_delta: { type: "number" },
        option_quantity: { type: "number" }
      },
      required: ["option_delta", "option_quantity"]
    }
  },
  {
    name: "options_breakeven",
    description: "Option Breakeven Points",
    inputSchema: {
      type: "object",
      properties: {
        strike: { type: "number" },
        premium: { type: "number" },
        is_call: { type: "boolean" }
      },
      required: ["strike", "premium", "is_call"]
    }
  }
];
var allTools = [
  ...movingAverageTools,
  ...momentumTools,
  ...volatilityTools,
  ...riskTools,
  ...portfolioTools,
  ...executionTools,
  ...regimeTools,
  ...conformalTools,
  ...greeksTools
];
var toolCategories = {
  moving_averages: movingAverageTools.length,
  momentum: momentumTools.length,
  volatility: volatilityTools.length,
  risk: riskTools.length,
  portfolio: portfolioTools.length,
  execution: executionTools.length,
  regime: regimeTools.length,
  conformal: conformalTools.length,
  greeks: greeksTools.length
};
var totalToolCount = allTools.length;

// src/index.ts
var __dirname2 = dirname(fileURLToPath(import.meta.url));
var projectRoot = resolve(__dirname2, "..");
var native = null;
var nativePaths = [
  process.env.NAUTILUS_NATIVE_PATH,
  resolve(projectRoot, "native/nautilus-native.darwin-x64.node"),
  resolve(projectRoot, "native/nautilus-native.darwin-arm64.node"),
  resolve(projectRoot, "native/target/release/libnautilus_native.dylib"),
  resolve(projectRoot, "dist/libnautilus_native.dylib")
];
for (const path of nativePaths) {
  if (path && existsSync(path)) {
    try {
      native = __require(path);
      console.error(`[Nautilus MCP] Loaded native module from ${path}`);
      break;
    } catch (e) {
      console.error(`[Nautilus MCP] Failed to load ${path}: ${e}`);
    }
  }
}
if (!native) {
  console.error("[Nautilus MCP] Warning: Native module not available, using JS fallback");
}
var fallback = {
  indicator_sma: (prices, period) => {
    const p = prices;
    const n = period;
    if (p.length < n || n === 0)
      return { success: false, error: "Insufficient data for SMA" };
    const sum = p.slice(-n).reduce((a, b) => a + b, 0);
    return { success: true, value: sum / n };
  },
  indicator_ema: (prices, period) => {
    const p = prices;
    const n = period;
    if (p.length === 0 || n === 0)
      return { success: false, error: "Insufficient data for EMA" };
    const multiplier = 2 / (n + 1);
    let ema = p[0];
    for (let i = 1;i < p.length; i++) {
      ema = (p[i] - ema) * multiplier + ema;
    }
    return { success: true, value: ema };
  },
  indicator_rsi: (prices, period) => {
    const p = prices;
    const n = period;
    if (p.length < n + 1)
      return { success: false, error: "Insufficient data for RSI" };
    let gains = 0, losses = 0;
    for (let i = p.length - n;i < p.length; i++) {
      const change = p[i] - p[i - 1];
      if (change > 0)
        gains += change;
      else
        losses -= change;
    }
    const avgGain = gains / n;
    const avgLoss = losses / n;
    if (avgLoss === 0)
      return { success: true, value: 100 };
    const rs = avgGain / avgLoss;
    return { success: true, value: 100 - 100 / (1 + rs) };
  },
  risk_var_parametric: (returns, confidence) => {
    const r = returns;
    const c = confidence;
    if (r.length === 0)
      return { success: false, error: "No returns" };
    const mean = r.reduce((a, b) => a + b, 0) / r.length;
    const variance = r.reduce((sum, x) => sum + (x - mean) ** 2, 0) / r.length;
    const std = Math.sqrt(variance);
    const z = c === 0.99 ? 2.326 : c === 0.95 ? 1.645 : 1.282;
    return { success: true, value: -(mean - z * std) };
  },
  portfolio_sharpe: (returns, rf) => {
    const r = returns;
    const riskFree = rf || 0;
    if (r.length === 0)
      return { success: false, error: "No returns" };
    const mean = r.reduce((a, b) => a + b, 0) / r.length;
    const variance = r.reduce((sum, x) => sum + (x - mean) ** 2, 0) / r.length;
    const std = Math.sqrt(variance);
    if (std === 0)
      return { success: true, value: 0 };
    return { success: true, value: (mean - riskFree) / std };
  },
  risk_kelly_criterion: (win_rate, win_loss_ratio) => {
    const w = win_rate;
    const wl = win_loss_ratio;
    if (w <= 0 || w >= 1 || wl <= 0)
      return { success: false, error: "Invalid parameters" };
    const kelly = (w * wl - (1 - w)) / wl;
    return {
      success: true,
      data: JSON.stringify({
        kelly_fraction: kelly,
        half_kelly: kelly / 2,
        quarter_kelly: kelly / 4
      })
    };
  },
  regime_pbit_state: (signal, volatility, temperature) => {
    const s = signal;
    const v = volatility;
    const t = temperature;
    const effectiveTemp = t * Math.max(v, 0.1);
    const probBullish = 1 / (1 + Math.exp(-s / effectiveTemp));
    const state = probBullish > 0.6 ? "bullish" : probBullish < 0.4 ? "bearish" : "neutral";
    return {
      success: true,
      data: JSON.stringify({
        prob_bullish: probBullish,
        prob_bearish: 1 - probBullish,
        state,
        temperature: effectiveTemp
      })
    };
  }
};
async function handleToolCall(name, args) {
  if (native && typeof native[name] === "function") {
    try {
      const result = native[name](...Object.values(args));
      return JSON.stringify(result);
    } catch (e) {
      return JSON.stringify({ success: false, error: `Native error: ${e}` });
    }
  }
  if (fallback[name]) {
    try {
      const result = fallback[name](...Object.values(args));
      return JSON.stringify(result);
    } catch (e) {
      return JSON.stringify({ success: false, error: `Fallback error: ${e}` });
    }
  }
  return JSON.stringify({ success: false, error: `Unknown tool: ${name}. Native module may be required.` });
}
var server = new Server({
  name: "nautilus-mcp",
  version: "1.0.0"
}, {
  capabilities: {
    tools: {}
  }
});
server.setRequestHandler(ListToolsRequestSchema2, async () => {
  return { tools: allTools };
});
server.setRequestHandler(CallToolRequestSchema2, async (request) => {
  const { name, arguments: args } = request.params;
  try {
    const result = await handleToolCall(name, args);
    return {
      content: [{ type: "text", text: result }]
    };
  } catch (error) {
    return {
      content: [{ type: "text", text: JSON.stringify({ success: false, error: String(error) }) }],
      isError: true
    };
  }
});
async function main() {
  console.error("\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557");
  console.error("\u2551              NAUTILUS MCP SERVER v1.0                        \u2551");
  console.error("\u2551       High-Performance Trading Analytics Platform            \u2551");
  console.error("\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D");
  console.error("");
  console.error(`  Native Module: ${native ? "\u2713 Loaded" : "\u2717 Using fallback"}`);
  console.error(`  Tools Available: ${totalToolCount}`);
  console.error(`  Categories:`);
  for (const [category, count] of Object.entries(toolCategories)) {
    console.error(`    - ${category}: ${count}`);
  }
  console.error("");
  if (!native) {
    console.error("  \u26A0\uFE0F  WARNING: Running without native module");
    console.error("  \u26A0\uFE0F  Some tools require native module for full functionality");
    console.error("  \u26A0\uFE0F  Build with: bun run build:native");
    console.error("");
  }
  const transport = new StdioServerTransport;
  await server.connect(transport);
  console.error("  [Ready] Listening on stdio transport");
}
main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
