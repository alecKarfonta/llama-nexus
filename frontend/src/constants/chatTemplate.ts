/** GGUF embedded chat template with --jinja only (no --chat-template-file). */
export const TEMPLATE_EMBEDDED_JINJA = '__embedded_jinja__'

export function isEmbeddedJinjaTemplate(selected: string): boolean {
  return selected === TEMPLATE_EMBEDDED_JINJA
}
