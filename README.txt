===========
Grid Sim
===========

Grid Sim provides a Linear Program based solution for different types
of energy sources.  It mainly differs from other simulators in that it
uses actual hour-by-hourprofile data for non-dispatchable sources,
such as wind and solar.  This allows for more reliable simulation than
simple LCOE (Levelized Cost of Energy) simulations which assume the
fungibility of energy.  E.g. An LCOE analysis wouldn't consider that
solar power cannot provide power in the darkest of night.

Example usage often looks like::

    #!/usr/bin/env python

    from towelstuff import location
    from towelstuff import utils

    if utils.has_towel():
        print "Your towel is located:", location.where_is_my_towel()

(Note the double-colon and 4-space indent formatting above.)

Paragraphs are separated by blank lines. *Italics*, **bold**,
and ``monospace`` look like this.


A Section
=========

Lists look like this:

* First

* Second. Can be multiple lines
  but must be indented properly.

A Sub-Section
-------------

Numbered lists look like you'd expect:

1. hi there

2. must be going

Urls are http://like.this and links can be
written `like this <http://www.example.com/foo/bar>`_.
