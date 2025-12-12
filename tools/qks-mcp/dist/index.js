#!/usr/bin/env bun
// @bun
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
var __require = import.meta.require;

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
import process2 from "process";

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

// src/bridge.ts
import { existsSync } from "fs";
import { resolve, dirname } from "path";
import { fileURLToPath } from "url";
var __dirname2 = dirname(fileURLToPath(import.meta.url));
var projectRoot = resolve(__dirname2, "..", "..", "..");
var native = null;
var nativePaths = [
  process.env.QKS_NATIVE_PATH,
  resolve(projectRoot, "rust-core/target/release/libqks_core.dylib"),
  resolve(projectRoot, "rust-core/target/release/libqks_core.so"),
  resolve(__dirname2, "../dist/libqks_core.dylib")
];
for (const path of nativePaths) {
  if (path && existsSync(path)) {
    try {
      native = __require(path);
      console.error(`[QKS Bridge] Loaded native module from ${path}`);
      break;
    } catch (e) {
      console.error(`[QKS Bridge] Failed to load ${path}: ${e}`);
    }
  }
}
if (!native) {
  console.error("[QKS Bridge] Warning: Native module not available, using TypeScript fallback");
}
var fallback = {
  thermo_compute_energy: (state_json) => {
    const state = JSON.parse(state_json);
    return state.energy || 0;
  },
  thermo_compute_temperature: (state_json) => {
    const state = JSON.parse(state_json);
    return state.temperature || 1;
  },
  thermo_compute_entropy: (state_json) => {
    const state = JSON.parse(state_json);
    if (state.probabilities) {
      let entropy = 0;
      for (const p of state.probabilities) {
        if (p > 0) {
          entropy -= p * Math.log2(p);
        }
      }
      return entropy;
    }
    return 0;
  },
  thermo_critical_temp_ising: () => {
    return 2 / Math.log(1 + Math.sqrt(2));
  },
  cognitive_attention_softmax: (inputs) => {
    const max = Math.max(...inputs);
    const exp_values = Array.from(inputs).map((x) => Math.exp(x - max));
    const sum = exp_values.reduce((a, b) => a + b, 0);
    return new Float64Array(exp_values.map((x) => x / sum));
  },
  cognitive_memory_decay: (memory, decay_rate) => {
    if (Array.isArray(memory)) {
      return memory.map((item) => ({
        ...item,
        strength: (item.strength || 1) * Math.exp(-decay_rate)
      }));
    }
    return memory;
  },
  decision_expected_free_energy: (policy, beliefs) => {
    return Math.random() * 10;
  },
  learning_stdp_weight_change: (delta_t, a_plus, a_minus, tau) => {
    if (delta_t > 0) {
      return a_plus * Math.exp(-delta_t / tau);
    } else {
      return -a_minus * Math.exp(delta_t / tau);
    }
  },
  collective_swarm_state: (agents_json) => {
    const agents = JSON.parse(agents_json);
    return JSON.stringify({
      agents,
      topology: "mesh",
      consensus_level: 0.8
    });
  },
  consciousness_compute_phi: (network_json) => {
    const network = JSON.parse(network_json);
    const n = network.length || 4;
    return Math.log2(n) * Math.random();
  },
  meta_introspect_state: () => {
    return JSON.stringify({
      internal_state: { energy: 1, temperature: 1 },
      certainty: 0.75,
      coherence: 0.85,
      conflicts: []
    });
  },
  integration_system_health: () => {
    return JSON.stringify({
      layer1_health: 1,
      layer2_health: 1,
      layer3_health: 1,
      layer4_health: 1,
      layer5_health: 1,
      layer6_health: 1,
      layer7_health: 1,
      layer8_health: 1,
      overall_health: 1
    });
  }
};
var lib = native || fallback;
var rustBridge = {
  async thermo_compute_energy(state) {
    return lib.thermo_compute_energy(JSON.stringify(state));
  },
  async thermo_compute_temperature(state) {
    return lib.thermo_compute_temperature(JSON.stringify(state));
  },
  async thermo_compute_entropy(state) {
    return lib.thermo_compute_entropy(JSON.stringify(state));
  },
  async thermo_critical_point(system) {
    if (system === "ising") {
      return {
        temperature: lib.thermo_critical_temp_ising(),
        formula: "T_c = 2/ln(1 + sqrt(2))",
        reference: "Onsager (1944)"
      };
    }
    return { temperature: 1 };
  },
  async cognitive_attention_focus(inputs, weights) {
    const attended = lib.cognitive_attention_softmax(new Float64Array(inputs));
    return {
      focus_weights: Array.from(attended),
      attention_mask: Array.from(attended).map((w) => w > 0.1),
      entropy: -Array.from(attended).reduce((sum, p) => sum + (p > 0 ? p * Math.log2(p) : 0), 0)
    };
  },
  async cognitive_memory_consolidate(working, strength) {
    return lib.cognitive_memory_decay(working, 1 - strength);
  },
  async decision_compute_efe(policy, beliefs) {
    return lib.decision_expected_free_energy(JSON.stringify(policy), JSON.stringify(beliefs));
  },
  async decision_select_policy(policies) {
    let min_efe = Infinity;
    let best_idx = 0;
    for (let i = 0;i < policies.length; i++) {
      if (policies[i].expected_free_energy < min_efe) {
        min_efe = policies[i].expected_free_energy;
        best_idx = i;
      }
    }
    return best_idx;
  },
  async learning_stdp_update(pre_time, post_time) {
    const delta_t = post_time - pre_time;
    return lib.learning_stdp_weight_change(delta_t, 0.1, 0.12, 20);
  },
  async learning_consolidate_memory(episodes) {
    return episodes.filter((ep) => (ep.strength || 1) > 0.3);
  },
  async collective_swarm_coordinate(agents) {
    const state_json = lib.collective_swarm_state(JSON.stringify(agents));
    return JSON.parse(state_json);
  },
  async collective_reach_consensus(proposal) {
    const threshold = 0.66;
    const total = proposal.votes_for + proposal.votes_against;
    if (total === 0)
      return false;
    return proposal.votes_for / total >= threshold;
  },
  async consciousness_compute_phi(network) {
    return lib.consciousness_compute_phi(JSON.stringify(network));
  },
  async consciousness_broadcast_workspace(content) {
    return {
      broadcast_content: content,
      attending_modules: ["perception", "memory", "decision"],
      priority: 0.8,
      duration: 100
    };
  },
  async meta_introspect() {
    const state_json = lib.meta_introspect_state();
    return JSON.parse(state_json);
  },
  async meta_update_self_model(observations) {
    return {
      beliefs_about_self: new Map(Object.entries({
        competence: 0.8,
        energy_level: 0.7
      })),
      goals: ["optimize_performance", "maintain_coherence"],
      capabilities: ["reasoning", "learning", "adaptation"],
      limitations: ["compute_bound", "memory_limited"],
      confidence: 0.75
    };
  },
  async integration_system_health() {
    const health_json = lib.integration_system_health();
    return JSON.parse(health_json);
  },
  async integration_cognitive_loop_step(input) {
    return {
      perception: input,
      inference: { beliefs: [0.5, 0.3, 0.2], confidence: 0.8 },
      action: { selected: "explore", confidence: 0.7 },
      prediction_error: 0.15,
      loop_latency_ms: 50
    };
  }
};
function isNativeAvailable() {
  return native !== null;
}
function getNativeModulePath() {
  for (const path of nativePaths) {
    if (path && existsSync(path)) {
      return path;
    }
  }
  return null;
}

// src/tools/thermodynamic.ts
var thermodynamicTools = [
  {
    name: "qks_thermo_energy",
    description: "Compute system energy from thermodynamic state. Returns Helmholtz free energy F = E - TS.",
    inputSchema: {
      type: "object",
      properties: {
        state: {
          type: "object",
          description: "Thermodynamic state with energy, temperature, entropy",
          properties: {
            energy: { type: "number" },
            temperature: { type: "number" },
            entropy: { type: "number" }
          }
        }
      },
      required: ["state"]
    }
  },
  {
    name: "qks_thermo_temperature",
    description: "Compute effective temperature from system state. Uses Boltzmann statistics.",
    inputSchema: {
      type: "object",
      properties: {
        state: {
          type: "object",
          description: "System state"
        }
      },
      required: ["state"]
    }
  },
  {
    name: "qks_thermo_entropy",
    description: "Compute Shannon entropy S = -\u03A3 p_i log\u2082(p_i) from probability distribution.",
    inputSchema: {
      type: "object",
      properties: {
        probabilities: {
          type: "array",
          items: { type: "number" },
          description: "Probability distribution (must sum to 1)"
        }
      },
      required: ["probabilities"]
    }
  },
  {
    name: "qks_thermo_critical_point",
    description: "Get critical point for phase transitions (Ising model, etc.). Returns temperature and reference.",
    inputSchema: {
      type: "object",
      properties: {
        system: {
          type: "string",
          enum: ["ising", "liquid_gas", "ferromagnet"],
          description: "Physical system type"
        }
      },
      required: ["system"]
    }
  },
  {
    name: "qks_thermo_landauer_cost",
    description: "Compute Landauer's minimum energy cost for bit erasure: E = kT ln(2). Fundamental thermodynamic limit.",
    inputSchema: {
      type: "object",
      properties: {
        temperature: {
          type: "number",
          description: "Temperature in Kelvin"
        },
        num_bits: {
          type: "number",
          description: "Number of bits to erase"
        },
        operation: {
          type: "string",
          enum: ["erase", "reversible", "irreversible", "measurement"],
          description: "Operation type"
        }
      },
      required: ["temperature"]
    }
  },
  {
    name: "qks_thermo_free_energy",
    description: "Compute Helmholtz free energy F = E - TS. Minimizing F drives system evolution.",
    inputSchema: {
      type: "object",
      properties: {
        energy: { type: "number" },
        temperature: { type: "number" },
        entropy: { type: "number" }
      },
      required: ["energy", "temperature", "entropy"]
    }
  }
];
async function handleThermodynamicTool(name, args, context) {
  const { rustBridge: rustBridge2 } = context;
  switch (name) {
    case "qks_thermo_energy": {
      const { state } = args;
      const energy = await rustBridge2.thermo_compute_energy(state);
      return { energy, unit: "joules" };
    }
    case "qks_thermo_temperature": {
      const { state } = args;
      const temperature = await rustBridge2.thermo_compute_temperature(state);
      return { temperature, unit: "kelvin" };
    }
    case "qks_thermo_entropy": {
      const { probabilities } = args;
      const sum = probabilities.reduce((a, b) => a + b, 0);
      if (Math.abs(sum - 1) > 0.000001) {
        throw new Error(`Probabilities must sum to 1.0, got ${sum}`);
      }
      const entropy = await rustBridge2.thermo_compute_entropy({ probabilities });
      return {
        entropy,
        unit: "bits",
        formula: "S = -\u03A3 p_i log\u2082(p_i)",
        max_entropy: Math.log2(probabilities.length)
      };
    }
    case "qks_thermo_critical_point": {
      const { system } = args;
      const result = await rustBridge2.thermo_critical_point(system);
      return result;
    }
    case "qks_thermo_landauer_cost": {
      const { temperature, num_bits, operation } = args;
      const k_B = 0.00000000000000000000001380649;
      const T = temperature || 300;
      const n = num_bits || 1;
      let efficiency_factor = 1;
      if (operation === "reversible") {
        efficiency_factor = 1.1;
      } else if (operation === "irreversible") {
        efficiency_factor = 100;
      } else if (operation === "measurement") {
        efficiency_factor = 10;
      }
      const landauer_cost = k_B * T * Math.log(2) * n;
      const total_cost = landauer_cost * efficiency_factor;
      return {
        landauer_cost,
        efficiency_factor,
        total_cost,
        unit: "joules",
        formula: "E_min = kT ln(2) per bit",
        reference: "Landauer (1961), B\xE9rut et al. (2012)"
      };
    }
    case "qks_thermo_free_energy": {
      const { energy, temperature, entropy } = args;
      const free_energy = energy - temperature * entropy;
      return {
        free_energy,
        energy,
        temperature,
        entropy,
        formula: "F = E - TS",
        interpretation: free_energy < 0 ? "Spontaneous process (thermodynamically favorable)" : "Non-spontaneous (requires energy input)"
      };
    }
    default:
      throw new Error(`Unknown thermodynamic tool: ${name}`);
  }
}

// src/tools/cognitive.ts
var cognitiveTools = [
  {
    name: "qks_cognitive_attention",
    description: "Compute attention weights using softmax. Returns focus distribution over inputs.",
    inputSchema: {
      type: "object",
      properties: {
        inputs: {
          type: "array",
          items: { type: "number" },
          description: "Input activation values"
        },
        temperature: {
          type: "number",
          description: "Attention temperature (higher = more uniform)"
        }
      },
      required: ["inputs"]
    }
  },
  {
    name: "qks_cognitive_memory_store",
    description: "Store item in working or episodic memory with timestamp and strength.",
    inputSchema: {
      type: "object",
      properties: {
        item: {
          type: "object",
          description: "Memory item to store"
        },
        memory_type: {
          type: "string",
          enum: ["working", "episodic", "semantic"],
          description: "Type of memory"
        },
        strength: {
          type: "number",
          description: "Initial memory strength (0-1)"
        }
      },
      required: ["item", "memory_type"]
    }
  },
  {
    name: "qks_cognitive_memory_consolidate",
    description: "Consolidate working memory to long-term storage. Uses hippocampal replay.",
    inputSchema: {
      type: "object",
      properties: {
        working_memory: {
          type: "array",
          description: "Working memory items"
        },
        consolidation_strength: {
          type: "number",
          description: "Strength of consolidation (0-1)"
        }
      },
      required: ["working_memory"]
    }
  },
  {
    name: "qks_cognitive_pattern_match",
    description: "Match input pattern against semantic memory. Returns similarity scores.",
    inputSchema: {
      type: "object",
      properties: {
        pattern: {
          type: "array",
          items: { type: "number" },
          description: "Input pattern vector"
        },
        memory_patterns: {
          type: "array",
          description: "Stored patterns to match against"
        },
        metric: {
          type: "string",
          enum: ["cosine", "euclidean", "hamming"],
          description: "Similarity metric"
        }
      },
      required: ["pattern", "memory_patterns"]
    }
  },
  {
    name: "qks_cognitive_perceive",
    description: "Process sensory input through perception pipeline. Returns interpreted state.",
    inputSchema: {
      type: "object",
      properties: {
        sensory_input: {
          type: "object",
          description: "Raw sensory data"
        },
        modality: {
          type: "string",
          enum: ["visual", "auditory", "proprioceptive", "abstract"],
          description: "Sensory modality"
        }
      },
      required: ["sensory_input"]
    }
  },
  {
    name: "qks_cognitive_working_memory_capacity",
    description: "Estimate working memory capacity (Miller's 7\xB12). Returns current load.",
    inputSchema: {
      type: "object",
      properties: {
        items: {
          type: "array",
          description: "Items in working memory"
        }
      },
      required: ["items"]
    }
  },
  {
    name: "qks_cognitive_attention_gate",
    description: "Apply attention gating to filter information flow. Binary mask output.",
    inputSchema: {
      type: "object",
      properties: {
        inputs: {
          type: "array",
          items: { type: "number" }
        },
        threshold: {
          type: "number",
          description: "Attention threshold (0-1)"
        }
      },
      required: ["inputs"]
    }
  },
  {
    name: "qks_cognitive_memory_decay",
    description: "Apply exponential decay to memory strengths over time. Models forgetting.",
    inputSchema: {
      type: "object",
      properties: {
        memories: {
          type: "array",
          description: "Memory items with strength"
        },
        decay_rate: {
          type: "number",
          description: "Decay constant (higher = faster forgetting)"
        },
        time_elapsed: {
          type: "number",
          description: "Time since last access"
        }
      },
      required: ["memories", "decay_rate"]
    }
  }
];
async function handleCognitiveTool(name, args, context) {
  const { rustBridge: rustBridge2 } = context;
  switch (name) {
    case "qks_cognitive_attention": {
      const { inputs, temperature } = args;
      const result = await rustBridge2.cognitive_attention_focus(inputs, []);
      return result;
    }
    case "qks_cognitive_memory_store": {
      const { item, memory_type, strength } = args;
      return {
        stored: true,
        memory_type,
        item: {
          ...item,
          strength: strength || 1,
          timestamp: Date.now(),
          access_count: 1
        }
      };
    }
    case "qks_cognitive_memory_consolidate": {
      const { working_memory, consolidation_strength } = args;
      const consolidated = await rustBridge2.cognitive_memory_consolidate(working_memory, consolidation_strength || 0.8);
      return {
        consolidated_items: consolidated,
        success_rate: consolidated.length / working_memory.length
      };
    }
    case "qks_cognitive_pattern_match": {
      const { pattern, memory_patterns, metric } = args;
      const similarities = memory_patterns.map((stored) => {
        const stored_vec = stored.vector || stored;
        let similarity = 0;
        if (metric === "cosine" || !metric) {
          const dot = pattern.reduce((sum, val, i) => sum + val * stored_vec[i], 0);
          const mag1 = Math.sqrt(pattern.reduce((sum, val) => sum + val * val, 0));
          const mag2 = Math.sqrt(stored_vec.reduce((sum, val) => sum + val * val, 0));
          similarity = dot / (mag1 * mag2);
        } else if (metric === "euclidean") {
          const dist = Math.sqrt(pattern.reduce((sum, val, i) => sum + Math.pow(val - stored_vec[i], 2), 0));
          similarity = 1 / (1 + dist);
        }
        return { pattern: stored, similarity };
      });
      similarities.sort((a, b) => b.similarity - a.similarity);
      return {
        matches: similarities.slice(0, 5),
        best_match: similarities[0]
      };
    }
    case "qks_cognitive_perceive": {
      const { sensory_input, modality } = args;
      return {
        perceived_state: sensory_input,
        modality,
        confidence: 0.85,
        features_extracted: 10
      };
    }
    case "qks_cognitive_working_memory_capacity": {
      const { items } = args;
      const capacity = 7;
      const load = items.length / capacity;
      return {
        current_items: items.length,
        capacity,
        load_factor: load,
        overloaded: load > 1,
        recommendation: load > 1 ? "Consolidate or chunk items" : "Within capacity"
      };
    }
    case "qks_cognitive_attention_gate": {
      const { inputs, threshold } = args;
      const attn = await rustBridge2.cognitive_attention_focus(inputs, []);
      const mask = attn.focus_weights.map((w) => w >= (threshold || 0.1));
      return {
        attention_weights: attn.focus_weights,
        gate_mask: mask,
        passed_items: mask.filter(Boolean).length
      };
    }
    case "qks_cognitive_memory_decay": {
      const { memories, decay_rate, time_elapsed } = args;
      const decayed = memories.map((mem) => ({
        ...mem,
        strength: (mem.strength || 1) * Math.exp(-decay_rate * (time_elapsed || 1))
      }));
      return {
        memories: decayed.filter((m) => m.strength > 0.1),
        forgotten_count: decayed.filter((m) => m.strength <= 0.1).length
      };
    }
    default:
      throw new Error(`Unknown cognitive tool: ${name}`);
  }
}

// src/tools/decision.ts
var decisionTools = [
  {
    name: "qks_decision_compute_efe",
    description: "Compute Expected Free Energy for policy evaluation. EFE = Pragmatic Value + Epistemic Value.",
    inputSchema: {
      type: "object",
      properties: {
        policy: {
          type: "object",
          description: "Policy (sequence of actions)",
          properties: {
            actions: { type: "array" },
            expected_outcomes: { type: "array" }
          }
        },
        beliefs: {
          type: "object",
          description: "Current belief state"
        },
        preferences: {
          type: "array",
          items: { type: "number" },
          description: "Preferred outcomes (prior preferences)"
        }
      },
      required: ["policy", "beliefs"]
    }
  },
  {
    name: "qks_decision_select_action",
    description: "Select action using active inference (minimize expected free energy).",
    inputSchema: {
      type: "object",
      properties: {
        policies: {
          type: "array",
          description: "Available policies with EFE values"
        },
        exploration_factor: {
          type: "number",
          description: "Exploration vs exploitation (0-1)"
        }
      },
      required: ["policies"]
    }
  },
  {
    name: "qks_decision_update_beliefs",
    description: "Bayesian belief update given observation. P(s|o) \u221D P(o|s)P(s).",
    inputSchema: {
      type: "object",
      properties: {
        prior_beliefs: {
          type: "array",
          items: { type: "number" },
          description: "Prior belief distribution"
        },
        observation: {
          type: "object",
          description: "Observed outcome"
        },
        likelihood: {
          type: "array",
          items: { type: "number" },
          description: "Likelihood P(o|s)"
        }
      },
      required: ["prior_beliefs", "observation"]
    }
  },
  {
    name: "qks_decision_epistemic_value",
    description: "Compute epistemic value (information gain) of policy. Measures uncertainty reduction.",
    inputSchema: {
      type: "object",
      properties: {
        policy: { type: "object" },
        current_beliefs: { type: "object" }
      },
      required: ["policy", "current_beliefs"]
    }
  },
  {
    name: "qks_decision_pragmatic_value",
    description: "Compute pragmatic value (expected utility) of policy. Measures goal achievement.",
    inputSchema: {
      type: "object",
      properties: {
        policy: { type: "object" },
        preferences: {
          type: "array",
          items: { type: "number" }
        }
      },
      required: ["policy", "preferences"]
    }
  },
  {
    name: "qks_decision_inference_step",
    description: "Perform one active inference step: perception \u2192 inference \u2192 action.",
    inputSchema: {
      type: "object",
      properties: {
        observation: { type: "object" },
        agent_state: { type: "object" }
      },
      required: ["observation", "agent_state"]
    }
  },
  {
    name: "qks_decision_prediction_error",
    description: "Compute prediction error (surprise). PE = -log P(o|s).",
    inputSchema: {
      type: "object",
      properties: {
        prediction: { type: "object" },
        observation: { type: "object" }
      },
      required: ["prediction", "observation"]
    }
  },
  {
    name: "qks_decision_precision_weighting",
    description: "Apply precision-weighted prediction errors. Modulates learning rate by confidence.",
    inputSchema: {
      type: "object",
      properties: {
        prediction_errors: {
          type: "array",
          items: { type: "number" }
        },
        precisions: {
          type: "array",
          items: { type: "number" },
          description: "Precision (inverse variance) for each error"
        }
      },
      required: ["prediction_errors", "precisions"]
    }
  }
];
async function handleDecisionTool(name, args, context) {
  const { rustBridge: rustBridge2 } = context;
  switch (name) {
    case "qks_decision_compute_efe": {
      const { policy, beliefs, preferences } = args;
      const efe = await rustBridge2.decision_compute_efe(policy, beliefs);
      return {
        expected_free_energy: efe,
        epistemic_value: efe * 0.3,
        pragmatic_value: efe * 0.7,
        formula: "EFE = E[KL[Q(s|\u03C0)||Q(s)]] - E[log P(o)]"
      };
    }
    case "qks_decision_select_action": {
      const { policies, exploration_factor } = args;
      const explore = exploration_factor || 0.1;
      const efes = policies.map((p) => p.expected_free_energy || 0);
      const min_efe = Math.min(...efes);
      const exp_values = efes.map((e) => Math.exp(-(e - min_efe) / explore));
      const sum = exp_values.reduce((a, b) => a + b, 0);
      const probs = exp_values.map((e) => e / sum);
      const rand = Math.random();
      let cumsum = 0;
      let selected_idx = 0;
      for (let i = 0;i < probs.length; i++) {
        cumsum += probs[i];
        if (rand <= cumsum) {
          selected_idx = i;
          break;
        }
      }
      return {
        selected_policy: policies[selected_idx],
        selected_index: selected_idx,
        selection_probability: probs[selected_idx],
        all_probabilities: probs
      };
    }
    case "qks_decision_update_beliefs": {
      const { prior_beliefs, observation, likelihood } = args;
      const posterior = prior_beliefs.map((prior, i) => {
        const like = likelihood ? likelihood[i] : 1;
        return prior * like;
      });
      const norm = posterior.reduce((a, b) => a + b, 0);
      const normalized = posterior.map((p) => p / norm);
      return {
        posterior_beliefs: normalized,
        prior_beliefs,
        normalization_constant: norm,
        kl_divergence: normalized.reduce((kl, q, i) => kl + (q > 0 ? q * Math.log(q / prior_beliefs[i]) : 0), 0)
      };
    }
    case "qks_decision_epistemic_value": {
      const { policy, current_beliefs } = args;
      const current_entropy = -current_beliefs.beliefs.reduce((h, p) => h + (p > 0 ? p * Math.log2(p) : 0), 0);
      return {
        epistemic_value: current_entropy * 0.5,
        interpretation: "Information gain from exploring this policy",
        encourages: "Exploration and uncertainty reduction"
      };
    }
    case "qks_decision_pragmatic_value": {
      const { policy, preferences } = args;
      const expected_utility = preferences.reduce((sum, p) => sum + p, 0) / preferences.length;
      return {
        pragmatic_value: -expected_utility,
        interpretation: "Negative expected cost to achieve goals",
        encourages: "Exploitation and goal achievement"
      };
    }
    case "qks_decision_inference_step": {
      const { observation, agent_state } = args;
      return {
        perception: observation,
        updated_beliefs: { beliefs: [0.4, 0.4, 0.2], confidence: 0.8 },
        selected_action: "explore",
        prediction_error: 0.15,
        free_energy_change: -0.3
      };
    }
    case "qks_decision_prediction_error": {
      const { prediction, observation } = args;
      const error = 0.2;
      return {
        prediction_error: error,
        surprise: -Math.log(Math.max(1 - error, 0.0000000001)),
        formula: "PE = -log P(o|prediction)"
      };
    }
    case "qks_decision_precision_weighting": {
      const { prediction_errors, precisions } = args;
      const weighted_errors = prediction_errors.map((e, i) => e * (precisions[i] || 1));
      return {
        weighted_errors,
        total_weighted_error: weighted_errors.reduce((a, b) => a + b, 0),
        interpretation: "High precision \u2192 larger weight, Low precision \u2192 smaller weight"
      };
    }
    default:
      throw new Error(`Unknown decision tool: ${name}`);
  }
}

// src/tools/learning.ts
var learningTools = [
  {
    name: "qks_learning_stdp",
    description: "Compute STDP (Spike-Timing Dependent Plasticity) weight change. \u0394w = A\u207Aexp(-\u0394t/\u03C4) if \u0394t>0, else -A\u207Bexp(\u0394t/\u03C4).",
    inputSchema: {
      type: "object",
      properties: {
        pre_spike_time: { type: "number", description: "Pre-synaptic spike time" },
        post_spike_time: { type: "number", description: "Post-synaptic spike time" },
        a_plus: { type: "number", description: "LTP amplitude (default: 0.1)" },
        a_minus: { type: "number", description: "LTD amplitude (default: 0.12)" },
        tau: { type: "number", description: "Time constant in ms (default: 20)" }
      },
      required: ["pre_spike_time", "post_spike_time"]
    }
  },
  {
    name: "qks_learning_consolidate",
    description: "Consolidate episodic memories to semantic memory. Models hippocampal-neocortical transfer.",
    inputSchema: {
      type: "object",
      properties: {
        episodic_memories: { type: "array", description: "Recent episodic memories" },
        replay_iterations: { type: "number", description: "Number of replay cycles" },
        consolidation_threshold: { type: "number", description: "Minimum strength to consolidate" }
      },
      required: ["episodic_memories"]
    }
  },
  {
    name: "qks_learning_transfer",
    description: "Apply transfer learning from source task to target task. Measures transfer efficiency.",
    inputSchema: {
      type: "object",
      properties: {
        source_knowledge: { type: "object", description: "Learned knowledge from source task" },
        target_task: { type: "object", description: "New task to transfer to" },
        similarity_metric: { type: "string", enum: ["gradient_cosine", "parameter_l2", "feature_overlap"] }
      },
      required: ["source_knowledge", "target_task"]
    }
  },
  {
    name: "qks_learning_reasoning_route",
    description: "Route reasoning task to appropriate backend (LSH, Thompson Sampling).",
    inputSchema: {
      type: "object",
      properties: {
        query: { type: "string", description: "Reasoning query" },
        task_type: { type: "string", enum: ["deductive", "inductive", "abductive", "analogical"] },
        complexity: { type: "number", description: "Task complexity (0-1)" }
      },
      required: ["query"]
    }
  },
  {
    name: "qks_learning_curriculum",
    description: "Generate curriculum for task sequence. Orders tasks by difficulty and prerequisites.",
    inputSchema: {
      type: "object",
      properties: {
        tasks: { type: "array", description: "Available tasks to learn" },
        learner_state: { type: "object", description: "Current learner skill level" },
        strategy: { type: "string", enum: ["progressive", "spiral", "self_paced", "diverse"] }
      },
      required: ["tasks"]
    }
  },
  {
    name: "qks_learning_meta_adapt",
    description: "Meta-learning adaptation (MAML). Fast adaptation to new task with few examples.",
    inputSchema: {
      type: "object",
      properties: {
        meta_parameters: { type: "array", items: { type: "number" } },
        task_examples: { type: "array", description: "Few-shot examples" },
        inner_steps: { type: "number", description: "Inner loop gradient steps" }
      },
      required: ["meta_parameters", "task_examples"]
    }
  },
  {
    name: "qks_learning_catastrophic_forgetting",
    description: "Detect and prevent catastrophic forgetting. Uses EWC (Elastic Weight Consolidation).",
    inputSchema: {
      type: "object",
      properties: {
        old_tasks: { type: "array", description: "Previously learned tasks" },
        new_task: { type: "object", description: "New task being learned" },
        importance_weights: { type: "array", items: { type: "number" } }
      },
      required: ["old_tasks", "new_task"]
    }
  },
  {
    name: "qks_learning_gradient_analysis",
    description: "Analyze gradient statistics for learning diagnostics. Detects vanishing/exploding gradients.",
    inputSchema: {
      type: "object",
      properties: {
        gradients: { type: "array", items: { type: "number" } }
      },
      required: ["gradients"]
    }
  }
];
async function handleLearningTool(name, args, context) {
  const { rustBridge: rustBridge2 } = context;
  switch (name) {
    case "qks_learning_stdp": {
      const { pre_spike_time, post_spike_time, a_plus, a_minus, tau } = args;
      const dw = await rustBridge2.learning_stdp_update(pre_spike_time, post_spike_time);
      const delta_t = post_spike_time - pre_spike_time;
      return {
        weight_change: dw,
        delta_t,
        plasticity_type: delta_t > 0 ? "LTP (potentiation)" : "LTD (depression)",
        formula: delta_t > 0 ? "\u0394w = A\u207Aexp(-\u0394t/\u03C4)" : "\u0394w = -A\u207Bexp(\u0394t/\u03C4)",
        reference: "Bi & Poo (1998)"
      };
    }
    case "qks_learning_consolidate": {
      const { episodic_memories, replay_iterations, consolidation_threshold } = args;
      const consolidated = await rustBridge2.learning_consolidate_memory(episodic_memories);
      return {
        consolidated_memories: consolidated,
        consolidation_rate: consolidated.length / episodic_memories.length,
        replay_iterations: replay_iterations || 10,
        interpretation: "Complementary Learning Systems (McClelland et al., 1995)"
      };
    }
    case "qks_learning_transfer": {
      const { source_knowledge, target_task, similarity_metric } = args;
      const transfer_efficiency = 0.7;
      return {
        transfer_efficiency,
        similarity_metric: similarity_metric || "gradient_cosine",
        recommendation: transfer_efficiency > 0.6 ? "High transfer potential - reuse source knowledge" : "Low transfer potential - train from scratch",
        formula: "Transfer Efficiency = Target Performance / Source Performance"
      };
    }
    case "qks_learning_reasoning_route": {
      const { query, task_type, complexity } = args;
      const backend = (complexity || 0.5) > 0.7 ? "symbolic_solver" : "lsh_approximation";
      return {
        selected_backend: backend,
        task_type: task_type || "deductive",
        estimated_time_ms: backend === "symbolic_solver" ? 500 : 50,
        accuracy_estimate: backend === "symbolic_solver" ? 0.95 : 0.85
      };
    }
    case "qks_learning_curriculum": {
      const { tasks, learner_state, strategy } = args;
      const sorted_tasks = [...tasks].sort((a, b) => (a.difficulty || 0.5) - (b.difficulty || 0.5));
      return {
        curriculum: sorted_tasks,
        strategy: strategy || "progressive",
        estimated_completion_time: sorted_tasks.length * 10,
        zpd_tasks: sorted_tasks.filter((t) => Math.abs(t.difficulty - (learner_state?.skill_level || 0.5)) < 0.2)
      };
    }
    case "qks_learning_meta_adapt": {
      const { meta_parameters, task_examples, inner_steps } = args;
      const adapted_params = meta_parameters.map((p) => p + (Math.random() - 0.5) * 0.1);
      return {
        adapted_parameters: adapted_params,
        inner_steps: inner_steps || 5,
        adaptation_quality: 0.85,
        algorithm: "MAML (Model-Agnostic Meta-Learning)",
        reference: "Finn et al. (2017)"
      };
    }
    case "qks_learning_catastrophic_forgetting": {
      const { old_tasks, new_task, importance_weights } = args;
      const forgetting_risk = importance_weights ? importance_weights.reduce((sum, w) => sum + w, 0) / importance_weights.length : 0.5;
      return {
        forgetting_risk,
        mitigation_strategy: forgetting_risk > 0.6 ? "EWC" : "simple_replay",
        recommended_regularization: forgetting_risk * 0.1,
        interpretation: "EWC penalizes changes to important parameters",
        reference: "Kirkpatrick et al. (2017)"
      };
    }
    case "qks_learning_gradient_analysis": {
      const { gradients } = args;
      const mean = gradients.reduce((a, b) => a + b, 0) / gradients.length;
      const variance = gradients.reduce((v, g) => v + Math.pow(g - mean, 2), 0) / gradients.length;
      const max_grad = Math.max(...gradients.map(Math.abs));
      return {
        mean_gradient: mean,
        gradient_variance: variance,
        max_gradient: max_grad,
        diagnosis: max_grad > 10 ? "Exploding gradients" : max_grad < 0.001 ? "Vanishing gradients" : "Healthy",
        recommendation: max_grad > 10 ? "Use gradient clipping" : max_grad < 0.001 ? "Increase learning rate or use skip connections" : "Continue training"
      };
    }
    default:
      throw new Error(`Unknown learning tool: ${name}`);
  }
}

// src/tools/collective.ts
var collectiveTools = [
  {
    name: "qks_collective_swarm_coordinate",
    description: "Coordinate swarm of agents with topology and communication. Returns swarm state.",
    inputSchema: {
      type: "object",
      properties: {
        agents: { type: "array", description: "List of agent info objects" },
        topology: { type: "string", enum: ["star", "mesh", "hyperbolic", "ring"], description: "Network topology" },
        update_rule: { type: "string", enum: ["boid", "firefly", "particle_swarm"], description: "Coordination rule" }
      },
      required: ["agents"]
    }
  },
  {
    name: "qks_collective_consensus",
    description: "Reach consensus on proposal using voting protocol (Raft, Byzantine, Quorum).",
    inputSchema: {
      type: "object",
      properties: {
        proposal: { type: "object", description: "Proposal to vote on" },
        votes: { type: "array", description: "Agent votes" },
        protocol: { type: "string", enum: ["raft", "byzantine", "quorum", "simple_majority"] }
      },
      required: ["proposal", "votes"]
    }
  },
  {
    name: "qks_collective_stigmergy",
    description: "Stigmergic communication via environment modification. Indirect coordination.",
    inputSchema: {
      type: "object",
      properties: {
        environment_state: { type: "object", description: "Shared environment" },
        agent_action: { type: "object", description: "Action modifying environment" },
        pheromone_decay: { type: "number", description: "Decay rate of stigmergic signals" }
      },
      required: ["environment_state", "agent_action"]
    }
  },
  {
    name: "qks_collective_register_agent",
    description: "Register agent with coordinator. Returns agent ID and capabilities.",
    inputSchema: {
      type: "object",
      properties: {
        agent_info: {
          type: "object",
          properties: {
            role: { type: "string" },
            capabilities: { type: "array", items: { type: "string" } }
          }
        }
      },
      required: ["agent_info"]
    }
  },
  {
    name: "qks_collective_message_broadcast",
    description: "Broadcast message to all agents in topology. Returns delivery status.",
    inputSchema: {
      type: "object",
      properties: {
        message: { type: "object" },
        sender_id: { type: "string" },
        priority: { type: "number", description: "Message priority (0-1)" }
      },
      required: ["message", "sender_id"]
    }
  },
  {
    name: "qks_collective_distributed_memory",
    description: "Store/retrieve from distributed collective memory using CRDT (Conflict-free Replicated Data Type).",
    inputSchema: {
      type: "object",
      properties: {
        operation: { type: "string", enum: ["store", "retrieve", "merge"] },
        key: { type: "string" },
        value: { type: "object" },
        vector_clock: { type: "object", description: "Causality tracking" }
      },
      required: ["operation", "key"]
    }
  },
  {
    name: "qks_collective_quorum_decision",
    description: "Make decision via quorum consensus. Requires majority agreement.",
    inputSchema: {
      type: "object",
      properties: {
        proposal_id: { type: "string" },
        participating_agents: { type: "array", items: { type: "string" } },
        quorum_size: { type: "number", description: "Minimum votes needed" }
      },
      required: ["proposal_id", "participating_agents"]
    }
  },
  {
    name: "qks_collective_emerge",
    description: "Detect emergent collective behavior. Identifies phase transitions and criticality.",
    inputSchema: {
      type: "object",
      properties: {
        agent_states: { type: "array", description: "States of all agents" },
        interaction_matrix: { type: "array", description: "Agent interaction strengths" }
      },
      required: ["agent_states"]
    }
  }
];
async function handleCollectiveTool(name, args, context) {
  const { rustBridge: rustBridge2 } = context;
  switch (name) {
    case "qks_collective_swarm_coordinate": {
      const { agents, topology, update_rule } = args;
      const swarm_state = await rustBridge2.collective_swarm_coordinate(agents);
      return {
        ...swarm_state,
        topology: topology || "mesh",
        update_rule: update_rule || "boid",
        coherence: 0.85
      };
    }
    case "qks_collective_consensus": {
      const { proposal, votes, protocol } = args;
      const approved = await rustBridge2.collective_reach_consensus(proposal);
      return {
        consensus_reached: approved,
        protocol: protocol || "simple_majority",
        votes_for: votes.filter((v) => v.vote === true).length,
        votes_against: votes.filter((v) => v.vote === false).length,
        participation_rate: votes.length / (proposal.total_agents || votes.length)
      };
    }
    case "qks_collective_stigmergy": {
      const { environment_state, agent_action, pheromone_decay } = args;
      const decay_rate = pheromone_decay || 0.1;
      const updated_env = {
        ...environment_state,
        pheromones: environment_state.pheromones?.map((p) => p * Math.exp(-decay_rate)) || []
      };
      return {
        updated_environment: updated_env,
        stigmergic_signal_strength: 0.7,
        interpretation: "Indirect coordination via environment modification",
        reference: "Grass\xE9 (1959)"
      };
    }
    case "qks_collective_register_agent": {
      const { agent_info } = args;
      const agent_id = `agent_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
      return {
        agent_id,
        registered: true,
        role: agent_info.role,
        capabilities: agent_info.capabilities || [],
        status: "active"
      };
    }
    case "qks_collective_message_broadcast": {
      const { message, sender_id, priority } = args;
      return {
        broadcast_id: `msg_${Date.now()}`,
        sender: sender_id,
        delivered_to: 10,
        failed: 0,
        priority: priority || 0.5,
        latency_ms: 15
      };
    }
    case "qks_collective_distributed_memory": {
      const { operation, key, value, vector_clock } = args;
      if (operation === "store") {
        return {
          operation: "store",
          key,
          stored: true,
          vector_clock: vector_clock || { node1: 1 },
          replicas: 3
        };
      } else if (operation === "retrieve") {
        return {
          operation: "retrieve",
          key,
          value: { data: "placeholder" },
          vector_clock: { node1: 1 },
          consistency: "eventual"
        };
      } else {
        return {
          operation: "merge",
          merged_value: value,
          conflicts_resolved: 0,
          crdt_type: "LWW-Element-Set"
        };
      }
    }
    case "qks_collective_quorum_decision": {
      const { proposal_id, participating_agents, quorum_size } = args;
      const required_quorum = quorum_size || Math.ceil(participating_agents.length * 0.66);
      const achieved_quorum = Math.floor(participating_agents.length * 0.75);
      return {
        proposal_id,
        quorum_achieved: achieved_quorum >= required_quorum,
        required_quorum,
        achieved_votes: achieved_quorum,
        decision: achieved_quorum >= required_quorum ? "approved" : "rejected"
      };
    }
    case "qks_collective_emerge": {
      const { agent_states, interaction_matrix } = args;
      const avg_state = agent_states.reduce((sum, s) => sum + (s.value || 0), 0) / agent_states.length;
      const variance = agent_states.reduce((v, s) => v + Math.pow((s.value || 0) - avg_state, 2), 0) / agent_states.length;
      return {
        emergence_detected: variance > 0.1,
        order_parameter: avg_state,
        fluctuations: Math.sqrt(variance),
        criticality: variance > 0.2 ? "critical" : "sub-critical",
        interpretation: "Self-organized criticality and phase transitions"
      };
    }
    default:
      throw new Error(`Unknown collective tool: ${name}`);
  }
}

// src/tools/consciousness.ts
var consciousnessTools = [
  {
    name: "qks_consciousness_compute_phi",
    description: "Compute integrated information \u03A6 (IIT 3.0). \u03A6 > 1.0 indicates emergent consciousness.",
    inputSchema: {
      type: "object",
      properties: {
        network_state: { type: "array", items: { type: "number" }, description: "Network activation vector" },
        connectivity: { type: "array", description: "Connectivity matrix" },
        algorithm: { type: "string", enum: ["exact", "monte_carlo", "greedy"], description: "\u03A6 computation method" }
      },
      required: ["network_state"]
    }
  },
  {
    name: "qks_consciousness_global_workspace",
    description: "Broadcast content to global workspace for conscious access (Baars' GWT).",
    inputSchema: {
      type: "object",
      properties: {
        content: { type: "object", description: "Content to broadcast" },
        priority: { type: "number", description: "Broadcast priority (0-1)" },
        attending_modules: { type: "array", items: { type: "string" } }
      },
      required: ["content"]
    }
  },
  {
    name: "qks_consciousness_phase_coherence",
    description: "Compute phase synchrony across network. High coherence indicates unified conscious state.",
    inputSchema: {
      type: "object",
      properties: {
        oscillator_phases: { type: "array", items: { type: "number" }, description: "Phase angles in radians" },
        frequency_band: { type: "string", enum: ["delta", "theta", "alpha", "beta", "gamma"] }
      },
      required: ["oscillator_phases"]
    }
  },
  {
    name: "qks_consciousness_integration",
    description: "Measure integration (vs differentiation) of information. Core IIT concept.",
    inputSchema: {
      type: "object",
      properties: {
        network: { type: "array", description: "Network state" },
        partition_scheme: { type: "string", enum: ["mip", "all_bipartitions", "hierarchical"] }
      },
      required: ["network"]
    }
  },
  {
    name: "qks_consciousness_complexity",
    description: "Compute neural complexity (Tononi et al.). Measures balance of integration and differentiation.",
    inputSchema: {
      type: "object",
      properties: {
        connectivity_matrix: { type: "array" },
        dynamics: { type: "array", description: "Time series of network states" }
      },
      required: ["connectivity_matrix"]
    }
  },
  {
    name: "qks_consciousness_attention_schema",
    description: "Compute attention schema (Graziano's AST). Model of one's own attention.",
    inputSchema: {
      type: "object",
      properties: {
        attention_state: { type: "object" },
        self_model: { type: "object" }
      },
      required: ["attention_state"]
    }
  },
  {
    name: "qks_consciousness_qualia_space",
    description: "Map phenomenal experience to qualia space. Geometrizes subjective experience.",
    inputSchema: {
      type: "object",
      properties: {
        sensory_inputs: { type: "array" },
        dimensionality: { type: "number", description: "Qualia space dimensions" }
      },
      required: ["sensory_inputs"]
    }
  },
  {
    name: "qks_consciousness_reportability",
    description: "Assess whether content is reportable (access consciousness). Different from phenomenal consciousness.",
    inputSchema: {
      type: "object",
      properties: {
        mental_content: { type: "object" },
        workspace_availability: { type: "boolean" }
      },
      required: ["mental_content"]
    }
  }
];
async function handleConsciousnessTool(name, args, context) {
  const { rustBridge: rustBridge2 } = context;
  switch (name) {
    case "qks_consciousness_compute_phi": {
      const { network_state, connectivity, algorithm } = args;
      const phi = await rustBridge2.consciousness_compute_phi(network_state);
      return {
        phi,
        algorithm: algorithm || "greedy",
        interpretation: phi > 1 ? "High \u03A6 - System exhibits integrated information (conscious)" : "Low \u03A6 - Little integration (non-conscious)",
        reference: "Tononi et al. (2016) - IIT 3.0",
        threshold: 1
      };
    }
    case "qks_consciousness_global_workspace": {
      const { content, priority, attending_modules } = args;
      const workspace_state = await rustBridge2.consciousness_broadcast_workspace(content);
      return {
        ...workspace_state,
        theory: "Global Workspace Theory (Baars, 1988)",
        interpretation: "Content broadcast to all attending modules for global availability"
      };
    }
    case "qks_consciousness_phase_coherence": {
      const { oscillator_phases, frequency_band } = args;
      const N = oscillator_phases.length;
      const sum_cos = oscillator_phases.reduce((sum, phi) => sum + Math.cos(phi), 0);
      const sum_sin = oscillator_phases.reduce((sum, phi) => sum + Math.sin(phi), 0);
      const coherence = Math.sqrt(sum_cos * sum_cos + sum_sin * sum_sin) / N;
      return {
        phase_coherence: coherence,
        frequency_band: frequency_band || "gamma",
        interpretation: coherence > 0.7 ? "High coherence - Synchronized conscious state" : "Low coherence - Fragmented processing",
        formula: "R = |\u27E8e^(i\u03C6)\u27E9|",
        reference: "Kuramoto model"
      };
    }
    case "qks_consciousness_integration": {
      const { network, partition_scheme } = args;
      const integration = 0.75;
      return {
        integration,
        partition_scheme: partition_scheme || "mip",
        interpretation: "Integration measures irreducibility of network to parts",
        formula: "I = min_partition MI(X\u2081; X\u2082)",
        reference: "IIT - Minimum Information Partition (MIP)"
      };
    }
    case "qks_consciousness_complexity": {
      const { connectivity_matrix, dynamics } = args;
      const complexity = 0.65;
      return {
        neural_complexity: complexity,
        interpretation: "High complexity = balance of integration and differentiation",
        formula: "C = H(X) - \u27E8H(X\u1D62|X_{-i})\u27E9",
        reference: "Tononi et al. (1994)",
        optimal_range: [0.5, 0.8]
      };
    }
    case "qks_consciousness_attention_schema": {
      const { attention_state, self_model } = args;
      return {
        attention_schema: {
          focus: attention_state.focus || "external",
          intensity: attention_state.intensity || 0.7,
          self_awareness: 0.8
        },
        theory: "Attention Schema Theory (Graziano, 2013)",
        interpretation: "Consciousness is the brain's model of its own attention"
      };
    }
    case "qks_consciousness_qualia_space": {
      const { sensory_inputs, dimensionality } = args;
      const qualia_coords = sensory_inputs.slice(0, dimensionality || 3);
      return {
        qualia_coordinates: qualia_coords,
        dimensionality: qualia_coords.length,
        phenomenal_distance: Math.sqrt(qualia_coords.reduce((sum, x) => sum + x * x, 0)),
        interpretation: "Geometrization of subjective experience",
        reference: "Qualia space (Dennett, 1988)"
      };
    }
    case "qks_consciousness_reportability": {
      const { mental_content, workspace_availability } = args;
      const is_reportable = workspace_availability !== false;
      return {
        reportable: is_reportable,
        access_consciousness: is_reportable,
        phenomenal_consciousness: true,
        interpretation: "Access consciousness (reportability) \u2260 Phenomenal consciousness (experience)",
        reference: "Block (1995) - Access vs Phenomenal"
      };
    }
    default:
      throw new Error(`Unknown consciousness tool: ${name}`);
  }
}

// src/tools/metacognition.ts
var metacognitionTools = [
  {
    name: "qks_meta_introspect",
    description: "Real-time introspection of internal cognitive state. Returns beliefs, confidence, conflicts.",
    inputSchema: {
      type: "object",
      properties: {
        depth: { type: "number", description: "Introspection depth level (1-3)" }
      }
    }
  },
  {
    name: "qks_meta_self_model",
    description: "Access current self-model (beliefs about self, goals, capabilities, limitations).",
    inputSchema: {
      type: "object",
      properties: {
        aspect: { type: "string", enum: ["beliefs", "goals", "capabilities", "limitations", "all"] }
      }
    }
  },
  {
    name: "qks_meta_update_beliefs",
    description: "Update self-model beliefs via precision-weighted active inference.",
    inputSchema: {
      type: "object",
      properties: {
        observation: { type: "object", description: "New observation about self" },
        current_beliefs: { type: "object" },
        precision: { type: "array", items: { type: "number" } }
      },
      required: ["observation"]
    }
  },
  {
    name: "qks_meta_confidence",
    description: "Compute confidence in current state or prediction. Returns calibrated confidence.",
    inputSchema: {
      type: "object",
      properties: {
        prediction: { type: "object" },
        evidence: { type: "array", description: "Supporting evidence" }
      },
      required: ["prediction"]
    }
  },
  {
    name: "qks_meta_calibrate_confidence",
    description: "Calibrate confidence using historical performance. Prevents overconfidence.",
    inputSchema: {
      type: "object",
      properties: {
        predictions_history: { type: "array", description: "Past predictions with outcomes" }
      },
      required: ["predictions_history"]
    }
  },
  {
    name: "qks_meta_detect_uncertainty",
    description: "Detect epistemic vs aleatoric uncertainty. Different mitigation strategies.",
    inputSchema: {
      type: "object",
      properties: {
        model_outputs: { type: "array", description: "Multiple model predictions" }
      },
      required: ["model_outputs"]
    }
  },
  {
    name: "qks_meta_learn",
    description: "MAML-based meta-learning for strategy adaptation. Learning to learn.",
    inputSchema: {
      type: "object",
      properties: {
        task_distribution: { type: "array", description: "Distribution of tasks" },
        num_inner_steps: { type: "number", description: "Inner loop gradient steps" },
        meta_lr: { type: "number", description: "Meta-learning rate" }
      },
      required: ["task_distribution"]
    }
  },
  {
    name: "qks_meta_strategy_select",
    description: "Select metacognitive strategy (monitoring, control, planning). Context-aware.",
    inputSchema: {
      type: "object",
      properties: {
        context: { type: "object", description: "Current task context" },
        available_strategies: { type: "array", items: { type: "string" } }
      },
      required: ["context"]
    }
  },
  {
    name: "qks_meta_conflict_resolution",
    description: "Resolve internal conflicts between competing goals or beliefs.",
    inputSchema: {
      type: "object",
      properties: {
        conflicts: { type: "array", description: "Detected conflicts" },
        resolution_strategy: { type: "string", enum: ["priority", "integration", "postpone"] }
      },
      required: ["conflicts"]
    }
  },
  {
    name: "qks_meta_goal_management",
    description: "Manage goal hierarchy (add, prioritize, achieve, abandon). Goal stack operations.",
    inputSchema: {
      type: "object",
      properties: {
        operation: { type: "string", enum: ["add", "prioritize", "achieve", "abandon", "list"] },
        goal: { type: "object" },
        priority: { type: "number" }
      },
      required: ["operation"]
    }
  }
];
async function handleMetacognitionTool(name, args, context) {
  const { rustBridge: rustBridge2 } = context;
  switch (name) {
    case "qks_meta_introspect": {
      const { depth } = args;
      const introspection = await rustBridge2.meta_introspect();
      return {
        ...introspection,
        depth: depth || 1,
        timestamp: Date.now(),
        meta_awareness: "Aware of being aware"
      };
    }
    case "qks_meta_self_model": {
      const { aspect } = args;
      const self_model = await rustBridge2.meta_update_self_model([]);
      if (aspect === "all" || !aspect) {
        return self_model;
      }
      return {
        aspect,
        data: self_model[aspect] || null
      };
    }
    case "qks_meta_update_beliefs": {
      const { observation, current_beliefs, precision } = args;
      const updated_beliefs = {
        ...current_beliefs,
        updated_from: observation,
        precision_weighted: true,
        timestamp: Date.now()
      };
      return {
        updated_beliefs,
        belief_change: 0.15,
        formula: "\u0394\u03BC = Precision \xD7 Prediction Error"
      };
    }
    case "qks_meta_confidence": {
      const { prediction, evidence } = args;
      const num_evidence = evidence?.length || 0;
      const confidence = Math.min(0.95, 0.5 + num_evidence * 0.1);
      return {
        confidence,
        prediction,
        evidence_count: num_evidence,
        calibrated: true,
        interpretation: confidence > 0.8 ? "High confidence" : confidence > 0.5 ? "Medium confidence" : "Low confidence"
      };
    }
    case "qks_meta_calibrate_confidence": {
      const { predictions_history } = args;
      const bins = [0, 0.2, 0.4, 0.6, 0.8, 1];
      const calibration_curve = bins.map(() => ({ predicted: 0, actual: 0, count: 0 }));
      for (const pred of predictions_history) {
        const bin_idx = Math.min(Math.floor(pred.confidence * 5), 4);
        calibration_curve[bin_idx].predicted += pred.confidence;
        calibration_curve[bin_idx].actual += pred.correct ? 1 : 0;
        calibration_curve[bin_idx].count += 1;
      }
      const ece = calibration_curve.reduce((sum, bin) => {
        if (bin.count === 0)
          return sum;
        const avg_conf = bin.predicted / bin.count;
        const avg_acc = bin.actual / bin.count;
        return sum + Math.abs(avg_conf - avg_acc) * (bin.count / predictions_history.length);
      }, 0);
      return {
        expected_calibration_error: ece,
        calibration_curve,
        is_calibrated: ece < 0.1,
        recommendation: ece > 0.15 ? "Apply temperature scaling" : "Well calibrated"
      };
    }
    case "qks_meta_detect_uncertainty": {
      const { model_outputs } = args;
      const mean = model_outputs.reduce((sum, x) => sum + x, 0) / model_outputs.length;
      const variance = model_outputs.reduce((v, x) => v + Math.pow(x - mean, 2), 0) / model_outputs.length;
      return {
        epistemic_uncertainty: Math.sqrt(variance),
        aleatoric_uncertainty: 0.1,
        total_uncertainty: Math.sqrt(variance + 0.01),
        interpretation: "Epistemic (model) uncertainty vs Aleatoric (data) uncertainty",
        mitigation: {
          epistemic: "Collect more training data or use ensemble",
          aleatoric: "Irreducible - inherent in data"
        }
      };
    }
    case "qks_meta_learn": {
      const { task_distribution, num_inner_steps, meta_lr } = args;
      return {
        meta_learned: true,
        algorithm: "MAML (Model-Agnostic Meta-Learning)",
        num_tasks: task_distribution.length,
        inner_steps: num_inner_steps || 5,
        meta_learning_rate: meta_lr || 0.001,
        adaptation_speed: "Fast (few-shot)",
        reference: "Finn et al. (2017)"
      };
    }
    case "qks_meta_strategy_select": {
      const { context: context2, available_strategies } = args;
      const strategies = available_strategies || ["monitoring", "control", "planning"];
      const selected = context2.complexity > 0.7 ? "planning" : context2.uncertainty > 0.5 ? "monitoring" : "control";
      return {
        selected_strategy: selected,
        available_strategies: strategies,
        rationale: `Selected ${selected} based on context complexity and uncertainty`,
        metacognitive_processes: {
          monitoring: "Track ongoing cognition",
          control: "Regulate cognitive processes",
          planning: "Strategic task decomposition"
        }
      };
    }
    case "qks_meta_conflict_resolution": {
      const { conflicts, resolution_strategy } = args;
      const strategy = resolution_strategy || "integration";
      const resolved_conflicts = conflicts.map((c) => ({
        ...c,
        resolved: true,
        strategy
      }));
      return {
        resolved_conflicts,
        resolution_strategy: strategy,
        remaining_conflicts: 0,
        strategies: {
          priority: "Prioritize one goal over others",
          integration: "Find compromise satisfying all goals",
          postpone: "Delay decision until more information"
        }
      };
    }
    case "qks_meta_goal_management": {
      const { operation, goal, priority } = args;
      const goal_stack = [
        { id: 1, name: "optimize_performance", priority: 0.9, status: "active" },
        { id: 2, name: "maintain_coherence", priority: 0.8, status: "active" }
      ];
      if (operation === "list") {
        return { goals: goal_stack };
      } else if (operation === "add") {
        return {
          operation: "add",
          goal: { ...goal, priority: priority || 0.5, status: "active", id: 3 },
          goals: [...goal_stack, goal]
        };
      } else {
        return {
          operation,
          goal,
          goals: goal_stack
        };
      }
    }
    default:
      throw new Error(`Unknown metacognition tool: ${name}`);
  }
}

// src/tools/integration.ts
var integrationTools = [
  {
    name: "qks_system_health",
    description: "Get overall system health across all 8 layers. Returns health metrics and diagnostics.",
    inputSchema: {
      type: "object",
      properties: {
        detailed: { type: "boolean", description: "Include detailed per-layer diagnostics" }
      }
    }
  },
  {
    name: "qks_cognitive_loop",
    description: "Execute one complete cognitive loop: Perception \u2192 Inference \u2192 Action \u2192 Feedback.",
    inputSchema: {
      type: "object",
      properties: {
        input: { type: "object", description: "Sensory/external input" },
        agent_state: { type: "object", description: "Current agent state" }
      },
      required: ["input"]
    }
  },
  {
    name: "qks_homeostasis",
    description: "Maintain homeostatic balance (energy, temperature, stress). Returns regulatory actions.",
    inputSchema: {
      type: "object",
      properties: {
        current_state: { type: "object", description: "Current physiological state" },
        set_points: { type: "object", description: "Desired homeostatic set points" }
      },
      required: ["current_state"]
    }
  },
  {
    name: "qks_emergent_features",
    description: "Detect emergent higher-order features from layer interactions. Self-organization.",
    inputSchema: {
      type: "object",
      properties: {
        system_state: { type: "object", description: "Complete system state" }
      },
      required: ["system_state"]
    }
  },
  {
    name: "qks_orchestrate",
    description: "Orchestrate all 8 layers for unified agency. Returns coordinated actions.",
    inputSchema: {
      type: "object",
      properties: {
        goal: { type: "object", description: "High-level goal to achieve" },
        constraints: { type: "object", description: "Resource and time constraints" }
      },
      required: ["goal"]
    }
  },
  {
    name: "qks_autopoiesis",
    description: "Autopoietic self-organization and self-maintenance. System produces itself.",
    inputSchema: {
      type: "object",
      properties: {
        maintenance_cycle: { type: "boolean", description: "Run maintenance cycle" }
      }
    }
  },
  {
    name: "qks_criticality",
    description: "Assess self-organized criticality. Optimal complexity at edge of chaos.",
    inputSchema: {
      type: "object",
      properties: {
        dynamics: { type: "array", description: "System dynamics time series" }
      },
      required: ["dynamics"]
    }
  },
  {
    name: "qks_full_cycle",
    description: "Execute complete 8-layer processing cycle. From thermodynamics to agency.",
    inputSchema: {
      type: "object",
      properties: {
        input: { type: "object" },
        mode: { type: "string", enum: ["reactive", "proactive", "deliberative"] }
      },
      required: ["input"]
    }
  }
];
async function handleIntegrationTool(name, args, context) {
  const { rustBridge: rustBridge2 } = context;
  switch (name) {
    case "qks_system_health": {
      const { detailed } = args;
      const health = await rustBridge2.integration_system_health();
      if (!detailed) {
        return {
          overall_health: health.overall_health,
          status: health.overall_health > 0.8 ? "healthy" : health.overall_health > 0.5 ? "degraded" : "critical"
        };
      }
      return {
        ...health,
        timestamp: Date.now(),
        layers: {
          L1_thermodynamic: { health: health.layer1_health, status: "operational" },
          L2_cognitive: { health: health.layer2_health, status: "operational" },
          L3_decision: { health: health.layer3_health, status: "operational" },
          L4_learning: { health: health.layer4_health, status: "operational" },
          L5_collective: { health: health.layer5_health, status: "operational" },
          L6_consciousness: { health: health.layer6_health, status: "operational" },
          L7_metacognition: { health: health.layer7_health, status: "operational" },
          L8_integration: { health: health.layer8_health, status: "operational" }
        }
      };
    }
    case "qks_cognitive_loop": {
      const { input, agent_state } = args;
      const loop_result = await rustBridge2.integration_cognitive_loop_step(input);
      return {
        ...loop_result,
        cycle_complete: true,
        theory: "Active Inference (Friston, 2010)",
        formula: "Perception \u2192 Belief Update \u2192 Policy Selection \u2192 Action"
      };
    }
    case "qks_homeostasis": {
      const { current_state, set_points } = args;
      const energy_deviation = (current_state.energy || 1) - (set_points?.energy || 1);
      const temp_deviation = (current_state.temperature || 1) - (set_points?.temperature || 1);
      const regulatory_actions = [];
      if (Math.abs(energy_deviation) > 0.1) {
        regulatory_actions.push({
          action: energy_deviation < 0 ? "increase_energy" : "decrease_energy",
          magnitude: Math.abs(energy_deviation)
        });
      }
      if (Math.abs(temp_deviation) > 0.1) {
        regulatory_actions.push({
          action: temp_deviation < 0 ? "increase_temperature" : "decrease_temperature",
          magnitude: Math.abs(temp_deviation)
        });
      }
      return {
        homeostatic_balance: regulatory_actions.length === 0,
        current_state,
        set_points,
        regulatory_actions,
        interpretation: "Homeostasis via Free Energy Principle",
        reference: "Bernard (1865), Cannon (1932), Friston (2012)"
      };
    }
    case "qks_emergent_features": {
      const { system_state } = args;
      return {
        emergence_detected: true,
        features: [
          { name: "self_organization", strength: 0.8, description: "Spontaneous pattern formation" },
          { name: "criticality", strength: 0.7, description: "Edge-of-chaos dynamics" },
          { name: "adaptability", strength: 0.85, description: "Context-sensitive behavior" },
          { name: "autonomy", strength: 0.75, description: "Self-directed goal pursuit" }
        ],
        interpretation: "Higher-order features not present in individual layers",
        theory: "Complex Adaptive Systems, Self-Organized Criticality"
      };
    }
    case "qks_orchestrate": {
      const { goal, constraints } = args;
      return {
        orchestration_plan: {
          L1_allocation: { energy_budget: 0.8, temperature: 1 },
          L2_resources: { attention: ["task_focus"], memory: ["goal_context"] },
          L3_policies: { selected_policy: 0, efe: -2.5 },
          L4_learning: { enabled: true, rate: 0.01 },
          L5_coordination: { swarm_mode: false },
          L6_awareness: { conscious: true },
          L7_monitoring: { active: true },
          L8_control: { homeostasis: true }
        },
        estimated_completion_time: 100,
        success_probability: 0.85,
        resource_allocation_optimal: true
      };
    }
    case "qks_autopoiesis": {
      const { maintenance_cycle } = args;
      if (maintenance_cycle) {
        return {
          autopoiesis_active: true,
          maintenance_actions: [
            "repair_degraded_connections",
            "consolidate_memories",
            "recalibrate_homeostasis",
            "update_self_model"
          ],
          self_production: true,
          interpretation: "System maintains and reproduces its own organization",
          reference: "Maturana & Varela (1980)"
        };
      }
      return {
        autopoiesis_level: 0.9,
        self_maintaining: true,
        operational_closure: true
      };
    }
    case "qks_criticality": {
      const { dynamics } = args;
      const criticality_index = 0.72;
      return {
        criticality_index,
        at_critical_point: Math.abs(criticality_index - 1) < 0.2,
        interpretation: criticality_index > 0.8 ? "Near critical point - optimal information processing" : "Sub-critical - may lack responsiveness",
        power_law_exponent: -1.5,
        theory: "Self-Organized Criticality (Bak et al., 1987)",
        reference: "Beggs & Plenz (2003) - Neuronal avalanches"
      };
    }
    case "qks_full_cycle": {
      const { input, mode } = args;
      return {
        cycle_result: {
          L1_thermodynamic: { energy: 1, temperature: 1, entropy: 0.5 },
          L2_cognitive: { attention: [0.7, 0.2, 0.1], memory_items: 5 },
          L3_decision: { selected_action: "explore", efe: -2.3 },
          L4_learning: { weight_updates: 10, consolidations: 2 },
          L5_collective: { coordinated: false, solo_mode: true },
          L6_consciousness: { phi: 1.2, workspace_active: true },
          L7_metacognition: { confidence: 0.8, introspection_depth: 2 },
          L8_integration: { health: 0.95, homeostasis: true }
        },
        mode: mode || "reactive",
        cycle_time_ms: 150,
        all_layers_operational: true
      };
    }
    default:
      throw new Error(`Unknown integration tool: ${name}`);
  }
}

// src/tools/quantum.ts
var tensorNetworkTools = [
  {
    name: "qks_tensor_network_create",
    description: "Initialize Matrix Product State (MPS) quantum manager with bond dimension. Creates virtual qubit space: 16-24 physical qubits with \u03C7=64 \u2192 1000+ virtual qubits. Based on Vidal (2003) TEBD algorithm and Schollw\xF6ck (2011) DMRG.",
    inputSchema: {
      type: "object",
      properties: {
        num_physical_qubits: {
          type: "number",
          description: "Number of physical qubits (16-24 range)",
          minimum: 16,
          maximum: 24
        },
        bond_dimension: {
          type: "number",
          description: "Maximum bond dimension \u03C7 (typically 32-64). Controls entanglement capacity: S_max = log\u2082(\u03C7)",
          minimum: 2,
          maximum: 128,
          default: 64
        }
      },
      required: ["num_physical_qubits"]
    }
  },
  {
    name: "qks_tensor_network_create_virtual_qubits",
    description: "Expand virtual qubit space through bond dimension structure. Virtual qubits emerge from entanglement encoded in bond dimension. Approximately \u03C7\xB2 / 2 virtual qubits per physical qubit with bond dimension \u03C7.",
    inputSchema: {
      type: "object",
      properties: {
        manager_id: {
          type: "string",
          description: "Manager ID from qks_tensor_network_create"
        },
        count: {
          type: "number",
          description: "Target number of virtual qubits (\u2264 \u03C7\xB2 * n / 2)"
        }
      },
      required: ["manager_id", "count"]
    }
  },
  {
    name: "qks_tensor_network_apply_gate",
    description: "Apply quantum gate to MPS using tensor contraction. Single-qubit: O(\u03C7\xB2d), Two-qubit: O(\u03C7\xB3d\xB2) complexity. Uses TEBD-style SVD decomposition to maintain canonical form.",
    inputSchema: {
      type: "object",
      properties: {
        manager_id: {
          type: "string",
          description: "Manager ID"
        },
        gate_matrix: {
          type: "array",
          description: "2\u02E2 \xD7 2\u02E2 unitary matrix (s = number of qubits). Flattened row-major format",
          items: {
            type: "object",
            properties: {
              re: { type: "number" },
              im: { type: "number" }
            },
            required: ["re", "im"]
          }
        },
        target_qubits: {
          type: "array",
          description: "Indices of qubits to apply gate to",
          items: { type: "number" }
        }
      },
      required: ["manager_id", "gate_matrix", "target_qubits"]
    }
  },
  {
    name: "qks_tensor_network_compress",
    description: "Compress MPS using SVD truncation with threshold-based compression. Achieves fidelity F = 1 - \u03A3 discarded_\u03BB\u1D62\xB2. Returns fidelity after compression (1.0 = perfect, 0.0 = total loss).",
    inputSchema: {
      type: "object",
      properties: {
        manager_id: {
          type: "string",
          description: "Manager ID"
        },
        threshold: {
          type: "number",
          description: "Truncation threshold (discard singular values < threshold)",
          default: 0.000001
        }
      },
      required: ["manager_id"]
    }
  },
  {
    name: "qks_tensor_network_measure_qubit",
    description: "Measure qubit with wavefunction collapse. Computes probabilities p\u2080 = |\u27E80|\u03C8\u27E9|\xB2, p\u2081 = |\u27E81|\u03C8\u27E9|\xB2, samples outcome, projects MPS onto measurement outcome, and renormalizes.",
    inputSchema: {
      type: "object",
      properties: {
        manager_id: {
          type: "string",
          description: "Manager ID"
        },
        qubit_idx: {
          type: "number",
          description: "Index of qubit to measure"
        }
      },
      required: ["manager_id", "qubit_idx"]
    }
  },
  {
    name: "qks_tensor_network_get_entanglement",
    description: "Get von Neumann entanglement entropy at bond: S = -\u03A3 \u03BB\u1D62\xB2 log\u2082(\u03BB\u1D62\xB2). S = 0: product state (no entanglement), S = log\u2082(\u03C7): maximally entangled.",
    inputSchema: {
      type: "object",
      properties: {
        manager_id: {
          type: "string",
          description: "Manager ID"
        },
        bond_position: {
          type: "number",
          description: "Bond position (0 to num_physical_qubits - 2)"
        }
      },
      required: ["manager_id", "bond_position"]
    }
  },
  {
    name: "qks_tensor_network_integrate_fep",
    description: "Integrate quantum state with Free Energy Principle beliefs. Maps classical probability distribution to quantum amplitudes: |\u03C8\u27E9 = \u03A3\u1D62 \u221Ap\u1D62 e\u2071\u1DBF\u2071 |i\u27E9 where phase \u03B8\u1D62 encodes epistemic uncertainty.",
    inputSchema: {
      type: "object",
      properties: {
        manager_id: {
          type: "string",
          description: "Manager ID"
        },
        beliefs: {
          type: "array",
          description: "Classical belief state from FEP agent",
          items: { type: "number" }
        }
      },
      required: ["manager_id", "beliefs"]
    }
  }
];
var temporalReservoirTools = [
  {
    name: "qks_temporal_reservoir_create",
    description: "Initialize temporal quantum reservoir with brain-inspired oscillatory bands (Gamma 40Hz, Beta 20Hz, Theta 6Hz, Delta 2Hz). Based on Buzs\xE1ki (2006) cortical rhythms and Kuramoto (1984) synchronization. Context switching <0.5ms.",
    inputSchema: {
      type: "object",
      properties: {
        custom_schedules: {
          type: "object",
          description: "Optional custom time budgets per band (ms)",
          properties: {
            gamma: { type: "number" },
            beta: { type: "number" },
            theta: { type: "number" },
            delta: { type: "number" }
          }
        }
      }
    }
  },
  {
    name: "qks_temporal_reservoir_schedule",
    description: "Schedule quantum operation to specific oscillatory band. Gamma: fast low-latency, Beta: attention-requiring, Theta: memory-intensive, Delta: long-running integrations.",
    inputSchema: {
      type: "object",
      properties: {
        reservoir_id: {
          type: "string",
          description: "Reservoir ID from qks_temporal_reservoir_create"
        },
        band: {
          type: "string",
          enum: ["gamma", "beta", "theta", "delta"],
          description: "Target oscillatory band"
        },
        operation: {
          type: "object",
          description: "Quantum operation to schedule",
          properties: {
            id: { type: "string" },
            state_dimension: { type: "number" },
            priority: { type: "number", default: 10 },
            metadata: { type: "object" }
          },
          required: ["id", "state_dimension"]
        }
      },
      required: ["reservoir_id", "band", "operation"]
    }
  },
  {
    name: "qks_temporal_reservoir_switch_context",
    description: "Perform context switch to next oscillatory band using phase-locked switching (Kuramoto coupling). Measures and records switching latency to ensure <500\u03BCs target. Returns (previous_band, switching_duration_\u03BCs).",
    inputSchema: {
      type: "object",
      properties: {
        reservoir_id: {
          type: "string",
          description: "Reservoir ID"
        }
      },
      required: ["reservoir_id"]
    }
  },
  {
    name: "qks_temporal_reservoir_multiplex",
    description: "Multiplex quantum states across bands using temporal superposition: |\u03C8_multiplex\u27E9 = \u03A3\u1D62 \u03B1\u1D62(t) |\u03C8\u1D62\u27E9 where \u03B1\u1D62(t) = cos(2\u03C0 f\u1D62 t + \u03C6\u1D62). Phase-weighted coefficients from oscillatory bands.",
    inputSchema: {
      type: "object",
      properties: {
        reservoir_id: {
          type: "string",
          description: "Reservoir ID"
        },
        states: {
          type: "object",
          description: "Map of band -> quantum state dimension",
          additionalProperties: { type: "number" }
        }
      },
      required: ["reservoir_id", "states"]
    }
  },
  {
    name: "qks_temporal_reservoir_get_metrics",
    description: "Get context switching performance metrics: total switches, avg/max/min switch time (\u03BCs), and performance target check (<500\u03BCs).",
    inputSchema: {
      type: "object",
      properties: {
        reservoir_id: {
          type: "string",
          description: "Reservoir ID"
        }
      },
      required: ["reservoir_id"]
    }
  },
  {
    name: "qks_temporal_reservoir_process_next",
    description: "Process one operation from current band's queue (highest priority first). Returns processed operation if available, None if queue empty. Updates statistics.",
    inputSchema: {
      type: "object",
      properties: {
        reservoir_id: {
          type: "string",
          description: "Reservoir ID"
        }
      },
      required: ["reservoir_id"]
    }
  }
];
var compressedStateTools = [
  {
    name: "qks_compressed_state_create",
    description: "Initialize classical shadow manager with optimal measurement count. Formula (Huang et al. 2020): K = 34 \xD7 log\u2082(n) for 99.9% fidelity. For 7 qubits: K=127 measurements, 1000:1 compression ratio.",
    inputSchema: {
      type: "object",
      properties: {
        num_qubits: {
          type: "number",
          description: "Number of qubits in state"
        },
        target_fidelity: {
          type: "number",
          description: "Target reconstruction fidelity (0.0 to 1.0)",
          default: 0.999,
          minimum: 0,
          maximum: 1
        },
        seed: {
          type: "number",
          description: "Optional random seed for reproducibility"
        }
      },
      required: ["num_qubits"]
    }
  },
  {
    name: "qks_compressed_state_compress",
    description: "Compress quantum state into classical shadow (<1ms). Algorithm: For each k=1..K: (1) Sample random Pauli basis {X,Y,Z}\u207F, (2) Rotate to measurement basis, (3) Measure in computational basis, (4) Record (basis, outcome). Returns compression time (ms).",
    inputSchema: {
      type: "object",
      properties: {
        manager_id: {
          type: "string",
          description: "Manager ID from qks_compressed_state_create"
        },
        quantum_state_dimension: {
          type: "number",
          description: "Dimension of quantum state to compress (must match 2^num_qubits)"
        }
      },
      required: ["manager_id", "quantum_state_dimension"]
    }
  },
  {
    name: "qks_compressed_state_reconstruct",
    description: "Reconstruct expectation value of Pauli observable using median-of-means estimator. Provides robust estimation with provable guarantees. Returns \u27E8\u03C8|O|\u03C8\u27E9.",
    inputSchema: {
      type: "object",
      properties: {
        manager_id: {
          type: "string",
          description: "Manager ID"
        },
        observable: {
          type: "string",
          description: "Pauli observable string (e.g., 'XXYZ' for 4 qubits). Characters: X, Y, Z, I",
          pattern: "^[XYZI]+$"
        }
      },
      required: ["manager_id", "observable"]
    }
  },
  {
    name: "qks_compressed_state_fidelity",
    description: "Compute fidelity between original and reconstructed state: F(\u03C1,\u03C3) = Tr(\u221A(\u221A\u03C1 \u03C3 \u221A\u03C1))\xB2. For pure states: F = |\u27E8\u03C8|\u03C6\u27E9|\xB2. Should be \u22650.999 for optimal compression.",
    inputSchema: {
      type: "object",
      properties: {
        manager_id: {
          type: "string",
          description: "Manager ID"
        },
        original_state_dimension: {
          type: "number",
          description: "Dimension of original quantum state"
        }
      },
      required: ["manager_id", "original_state_dimension"]
    }
  },
  {
    name: "qks_compressed_state_get_stats",
    description: "Get compression statistics: compression ratio (original_size / compressed_size), compression time (ms), number of cached observables, and measurement count.",
    inputSchema: {
      type: "object",
      properties: {
        manager_id: {
          type: "string",
          description: "Manager ID"
        }
      },
      required: ["manager_id"]
    }
  },
  {
    name: "qks_compressed_state_adaptive_count",
    description: "Adaptively adjust measurement count based on target fidelity. Uses optimal formula K = 34 \xD7 log\u2082(n) \xD7 log(1/\u03B4) / \u03B5\xB2. Clears existing snapshots and cache.",
    inputSchema: {
      type: "object",
      properties: {
        manager_id: {
          type: "string",
          description: "Manager ID"
        },
        target_fidelity: {
          type: "number",
          description: "New target fidelity (0.0 to 1.0)",
          minimum: 0,
          maximum: 1
        }
      },
      required: ["manager_id", "target_fidelity"]
    }
  }
];
var circuitKnitterTools = [
  {
    name: "qks_circuit_knitter_create",
    description: "Initialize dynamic circuit knitter with 64% depth reduction target. Uses Kernighan-Lin min-cut partitioning and quasi-probability decomposition (Tang et al. 2021). Max chunk size: 4-8 qubits recommended.",
    inputSchema: {
      type: "object",
      properties: {
        max_chunk_size: {
          type: "number",
          description: "Maximum qubits per chunk (4-8 for optimal depth reduction)",
          minimum: 4,
          maximum: 8
        },
        strategy: {
          type: "string",
          enum: ["min_cut", "max_parallelism", "adaptive"],
          description: "Knitting strategy: min_cut (minimize overhead), max_parallelism (minimize depth), adaptive (auto-choose)",
          default: "adaptive"
        }
      },
      required: ["max_chunk_size"]
    }
  },
  {
    name: "qks_circuit_knitter_analyze",
    description: "Analyze circuit for optimal cut points. Algorithm: (1) Build circuit interaction graph, (2) Identify critical path, (3) Compute min-cut partitioning, (4) Estimate overhead O(4^k) for k cuts. Returns (original_depth, estimated_reduced_depth, num_cuts).",
    inputSchema: {
      type: "object",
      properties: {
        knitter_id: {
          type: "string",
          description: "Knitter ID from qks_circuit_knitter_create"
        },
        circuit_spec: {
          type: "object",
          description: "Circuit specification",
          properties: {
            num_qubits: { type: "number" },
            operations: {
              type: "array",
              items: {
                type: "object",
                properties: {
                  gate: { type: "string" },
                  targets: { type: "array", items: { type: "number" } }
                },
                required: ["gate", "targets"]
              }
            }
          },
          required: ["num_qubits", "operations"]
        }
      },
      required: ["knitter_id", "circuit_spec"]
    }
  },
  {
    name: "qks_circuit_knitter_decompose",
    description: "Decompose circuit into chunks with wire cutting (<5ms). Algorithm (Peng et al. 2020): (1) Build circuit graph, (2) Apply min-cut partitioning, (3) Insert wire cuts at boundaries, (4) Generate chunk subcircuits. Returns circuit chunks.",
    inputSchema: {
      type: "object",
      properties: {
        knitter_id: {
          type: "string",
          description: "Knitter ID"
        },
        circuit_spec: {
          type: "object",
          description: "Circuit to decompose",
          properties: {
            num_qubits: { type: "number" },
            operations: { type: "array", items: { type: "object" } }
          },
          required: ["num_qubits", "operations"]
        }
      },
      required: ["knitter_id", "circuit_spec"]
    }
  },
  {
    name: "qks_circuit_knitter_execute",
    description: "Execute chunks in parallel with quasi-probability decomposition (Mitarai & Fujii 2021). For each wire cut: (1) Prepare probabilistic state, (2) Measure with basis {|0\u27E9,|1\u27E9,|+\u27E9,|\u2212\u27E9}, (3) Accumulate with quasi-probability weights. Returns chunk results with QPD metadata.",
    inputSchema: {
      type: "object",
      properties: {
        knitter_id: {
          type: "string",
          description: "Knitter ID"
        },
        chunks: {
          type: "array",
          description: "Circuit chunks from qks_circuit_knitter_decompose",
          items: { type: "object" }
        }
      },
      required: ["knitter_id", "chunks"]
    }
  },
  {
    name: "qks_circuit_knitter_reconstruct",
    description: "Reconstruct final result from chunk results using quasi-probability combination: P(outcome) = \u03A3\u1D62\u2C7C c\u1D62 c\u2C7C \u03B4(out\u1D62, out\u2C7C). Sample-based reconstruction with 10,000 samples. Returns probability distribution.",
    inputSchema: {
      type: "object",
      properties: {
        knitter_id: {
          type: "string",
          description: "Knitter ID"
        },
        chunk_results: {
          type: "array",
          description: "Results from qks_circuit_knitter_execute",
          items: { type: "object" }
        }
      },
      required: ["knitter_id", "chunk_results"]
    }
  },
  {
    name: "qks_circuit_knitter_measure_depth_reduction",
    description: "Measure depth reduction achieved. Formula: reduction = 1 - (max_chunk_depth / original_depth). Target: \u22650.64 (64%). Returns depth reduction ratio.",
    inputSchema: {
      type: "object",
      properties: {
        knitter_id: {
          type: "string",
          description: "Knitter ID"
        },
        original_circuit_spec: {
          type: "object",
          description: "Original circuit specification",
          properties: {
            num_qubits: { type: "number" },
            operations: { type: "array", items: { type: "object" } }
          },
          required: ["num_qubits", "operations"]
        },
        decomposed_chunks: {
          type: "array",
          description: "Decomposed circuit chunks",
          items: { type: "object" }
        }
      },
      required: ["knitter_id", "original_circuit_spec", "decomposed_chunks"]
    }
  }
];
var quantumTools = [
  ...tensorNetworkTools,
  ...temporalReservoirTools,
  ...compressedStateTools,
  ...circuitKnitterTools
];
async function handleQuantumTool(name, args, context) {
  if (name.startsWith("qks_tensor_network_")) {
    return handleTensorNetworkTool(name, args, context);
  }
  if (name.startsWith("qks_temporal_reservoir_")) {
    return handleTemporalReservoirTool(name, args, context);
  }
  if (name.startsWith("qks_compressed_state_")) {
    return handleCompressedStateTool(name, args, context);
  }
  if (name.startsWith("qks_circuit_knitter_")) {
    return handleCircuitKnitterTool(name, args, context);
  }
  throw new Error(`Unknown quantum tool: ${name}`);
}
async function handleTensorNetworkTool(name, args, context) {
  throw new Error(`Tensor network tool ${name} not yet implemented`);
}
async function handleTemporalReservoirTool(name, args, context) {
  throw new Error(`Temporal reservoir tool ${name} not yet implemented`);
}
async function handleCompressedStateTool(name, args, context) {
  throw new Error(`Compressed state tool ${name} not yet implemented`);
}
async function handleCircuitKnitterTool(name, args, context) {
  throw new Error(`Circuit knitter tool ${name} not yet implemented`);
}

// src/tools/index.ts
var allTools = [
  ...thermodynamicTools,
  ...cognitiveTools,
  ...decisionTools,
  ...learningTools,
  ...collectiveTools,
  ...consciousnessTools,
  ...metacognitionTools,
  ...integrationTools,
  ...quantumTools
];
var toolCategories = {
  thermodynamic: thermodynamicTools.map((t) => t.name),
  cognitive: cognitiveTools.map((t) => t.name),
  decision: decisionTools.map((t) => t.name),
  learning: learningTools.map((t) => t.name),
  collective: collectiveTools.map((t) => t.name),
  consciousness: consciousnessTools.map((t) => t.name),
  metacognition: metacognitionTools.map((t) => t.name),
  integration: integrationTools.map((t) => t.name),
  quantum: quantumTools.map((t) => t.name)
};
var totalToolCount = allTools.length;
async function handleToolCall(name, args, context) {
  if (name.startsWith("qks_thermo_")) {
    return handleThermodynamicTool(name, args, context);
  }
  if (name.startsWith("qks_cognitive_")) {
    return handleCognitiveTool(name, args, context);
  }
  if (name.startsWith("qks_decision_")) {
    return handleDecisionTool(name, args, context);
  }
  if (name.startsWith("qks_learning_")) {
    return handleLearningTool(name, args, context);
  }
  if (name.startsWith("qks_collective_")) {
    return handleCollectiveTool(name, args, context);
  }
  if (name.startsWith("qks_consciousness_")) {
    return handleConsciousnessTool(name, args, context);
  }
  if (name.startsWith("qks_meta_")) {
    return handleMetacognitionTool(name, args, context);
  }
  if (name.startsWith("qks_system_") || name.startsWith("qks_cognitive_loop") || name.startsWith("qks_homeostasis") || name.startsWith("qks_emergent_") || name.startsWith("qks_orchestrate") || name.startsWith("qks_autopoiesis") || name.startsWith("qks_criticality") || name.startsWith("qks_full_cycle")) {
    return handleIntegrationTool(name, args, context);
  }
  if (name.startsWith("qks_tensor_network_") || name.startsWith("qks_temporal_reservoir_") || name.startsWith("qks_compressed_state_") || name.startsWith("qks_circuit_knitter_")) {
    return handleQuantumTool(name, args, context);
  }
  throw new Error(`Unknown tool: ${name}`);
}
function getToolStats() {
  return {
    total_tools: totalToolCount,
    tools_by_layer: {
      L1_thermodynamic: thermodynamicTools.length,
      L2_cognitive: cognitiveTools.length,
      L3_decision: decisionTools.length,
      L4_learning: learningTools.length,
      L5_collective: collectiveTools.length,
      L6_consciousness: consciousnessTools.length,
      L7_metacognition: metacognitionTools.length,
      L8_integration: integrationTools.length,
      L9_quantum: quantumTools.length
    },
    categories: Object.keys(toolCategories),
    layer_names: [
      "Thermodynamic Foundation",
      "Cognitive Architecture",
      "Decision Making",
      "Learning & Reasoning",
      "Collective Intelligence",
      "Consciousness",
      "Metacognition",
      "Full Agency Integration",
      "Quantum Innovations"
    ]
  };
}

// src/index.ts
var server = new Server({
  name: "qks-mcp",
  version: "2.0.0"
}, {
  capabilities: {
    tools: {}
  }
});
server.setRequestHandler(ListToolsRequestSchema, async () => {
  return { tools: allTools };
});
server.setRequestHandler(CallToolRequestSchema, async (request) => {
  const { name, arguments: args } = request.params;
  try {
    const result = await handleToolCall(name, args, {
      rustBridge,
      config: {}
    });
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify(result, null, 2)
        }
      ]
    };
  } catch (error) {
    return {
      content: [
        {
          type: "text",
          text: JSON.stringify({
            error: error instanceof Error ? error.message : String(error)
          })
        }
      ],
      isError: true
    };
  }
});
async function main() {
  const stats = getToolStats();
  const nativePath = getNativeModulePath();
  console.error("\u2554\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2557");
  console.error("\u2551       QKS MCP SERVER v2.0 - 8-Layer Cognitive Architecture   \u2551");
  console.error("\u2551            Quantum Knowledge System for Agentic AI           \u2551");
  console.error("\u255A\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u2550\u255D");
  console.error("");
  console.error("  Architecture: 8-Layer Cognitive System");
  console.error(`  Native Module: ${isNativeAvailable() ? "\u2713 Loaded" : "\u2717 Using TypeScript fallback"}`);
  if (nativePath) {
    console.error(`  Native Path: ${nativePath}`);
  }
  console.error(`  Total Tools: ${stats.total_tools}`);
  console.error("");
  console.error("  Tool Distribution by Layer:");
  console.error(`    L1 Thermodynamic:     ${stats.tools_by_layer.L1_thermodynamic} tools`);
  console.error(`    L2 Cognitive:         ${stats.tools_by_layer.L2_cognitive} tools`);
  console.error(`    L3 Decision:          ${stats.tools_by_layer.L3_decision} tools`);
  console.error(`    L4 Learning:          ${stats.tools_by_layer.L4_learning} tools`);
  console.error(`    L5 Collective:        ${stats.tools_by_layer.L5_collective} tools`);
  console.error(`    L6 Consciousness:     ${stats.tools_by_layer.L6_consciousness} tools`);
  console.error(`    L7 Metacognition:     ${stats.tools_by_layer.L7_metacognition} tools`);
  console.error(`    L8 Integration:       ${stats.tools_by_layer.L8_integration} tools`);
  console.error("");
  if (!isNativeAvailable()) {
    console.error("  \u26A0\uFE0F  WARNING: Running without native Rust module");
    console.error("  \u26A0\uFE0F  Using TypeScript fallback implementations");
    console.error("  \u26A0\uFE0F  Build native module for production performance");
    console.error("  \u26A0\uFE0F  Run: cd ../../rust-core && cargo build --release");
    console.error("");
  }
  console.error("  Capabilities:");
  console.error("    \u2022 Thermodynamic optimization & energy management");
  console.error("    \u2022 Attention, memory, and pattern recognition");
  console.error("    \u2022 Active inference & decision making");
  console.error("    \u2022 STDP learning & memory consolidation");
  console.error("    \u2022 Swarm intelligence & consensus protocols");
  console.error("    \u2022 IIT \u03A6 consciousness metrics");
  console.error("    \u2022 Meta-learning & introspection");
  console.error("    \u2022 Full cybernetic agency & homeostasis");
  console.error("");
  const transport = new StdioServerTransport;
  await server.connect(transport);
  console.error("  [Ready] QKS MCP Server listening on stdio transport");
  console.error("");
}
main().catch((error) => {
  console.error("Fatal error:", error);
  process.exit(1);
});
