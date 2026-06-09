/** Build grammar / JSON-schema fields for chat completion requests. */

import type { ChatCompletionRequest } from '@/types/api';

export interface StructuredOutputSettings {
  enableStructuredOutput: boolean;
  structuredOutputSchema: Record<string, unknown> | null;
  structuredOutputGrammar: string | null;
  structuredOutputSchemaName?: string;
}

export function hasActiveStructuredOutput(settings: StructuredOutputSettings): boolean {
  return Boolean(settings.enableStructuredOutput && settings.structuredOutputSchema);
}

export function structuredOutputConflictsWithTools(
  settings: StructuredOutputSettings & { enableTools: boolean; selectedTools: string[] }
): boolean {
  return (
    hasActiveStructuredOutput(settings) &&
    settings.enableTools &&
    settings.selectedTools.length > 0
  );
}

/** OpenAI-compatible response_format + optional GBNF grammar for llama-server. */
export function buildChatOutputConstraints(
  settings: StructuredOutputSettings
): Pick<ChatCompletionRequest, 'response_format' | 'grammar'> {
  if (!hasActiveStructuredOutput(settings) || !settings.structuredOutputSchema) {
    return {};
  }

  const schema = settings.structuredOutputSchema;
  const title =
    settings.structuredOutputSchemaName ||
    (typeof schema.title === 'string' ? schema.title : 'response');

  const { $schema: _s, title: _t, ...schemaBody } = schema;
  const innerSchema =
    typeof schemaBody.type === 'string' ? schemaBody : schema;

  const result: Pick<ChatCompletionRequest, 'response_format' | 'grammar'> = {
    response_format: {
      type: 'json_schema',
      json_schema: {
        name: title,
        strict: true,
        schema: innerSchema,
      },
    },
  };

  const grammar = settings.structuredOutputGrammar?.trim();
  if (grammar) {
    result.grammar = grammar;
  }

  return result;
}
