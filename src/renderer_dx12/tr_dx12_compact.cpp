#include <cstring>
#include <cstdio>

#define MAX_INFO_STRING 1024

void Q_strncpyz( char* dest, const char* src, unsigned int destsize )
{
	if ( !dest || destsize <= 0 )
	{
		return;
	}
	if ( !src )
	{
		dest[ 0 ] = '\0';
		return;
	}

	std::strncpy( dest, src, destsize - 1 );
	dest[ destsize - 1 ] = '\0';
}

void Info_SetValueForKey( char* s, const char* key, const char* value )
{
	char newPair[ MAX_INFO_STRING ];

	std::snprintf( newPair, sizeof( newPair ), "\\%s\\%s", key, value );

	if ( std::strlen( s ) + std::strlen( newPair ) >= MAX_INFO_STRING )
	{
		return;
	}

	std::strcat( s, newPair );
}
