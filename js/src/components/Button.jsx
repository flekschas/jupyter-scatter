import { forwardRef } from 'react';

export const Button = forwardRef(function Button(
  { icon, children, tooltip, active, disabled, onClick, class: className },
  ref,
) {
  return (
    <button
      ref={ref}
      type="button"
      title={tooltip}
      disabled={disabled}
      onClick={onClick}
      class={[
        'jss:flex jss:items-center jss:justify-center',
        'jss:rounded-sm! jss:focus-ring',
        'jss:cursor-pointer',
        'jss:transition-all jss:ease-out jss:duration-150',
        active
          ? 'jss:bg-brand jss:text-white jss:hover:bg-brand-hover'
          : 'jss:bg-bg jss:text-fg jss:hover:bg-bg-hover',
        disabled
          ? 'jss:opacity-40 jss:pointer-events-none'
          : 'jss:active:scale-95',
        className ?? 'jss:w-8 jss:h-8 jss:p-1',
      ]
        .filter(Boolean)
        .join(' ')}
    >
      {icon}
      {children}
    </button>
  );
});
